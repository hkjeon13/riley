"""CPU-only predecessor-chain and unchanged-lifecycle checks; no process actions."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


m = module('round15_factory_test', HERE / 'prepare_remote_session_round15.py')
old = module('round14_factory_reference', HERE / 'remote_session_round14.py')


class ChainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.previous = self.root / 'blender-round14'
        self.previous.mkdir()
        self.context = [dict(pid=10+i, start=str(100+i), port=port, gui_env={'DISPLAY': ':1'})
                        for i, port in enumerate(old.PORTS)]
        self.proven = [dict(row, pid=20+i, start=str(200+i), commands_and_gui_environment_match=True)
                       for i, row in enumerate(self.context)]
        self.before = [dict(pid=row['pid'], start=row['start'], port=row['port'], env=row['gui_env'],
                            argv=['blender', str(i)+'.blend'], cwd='/models', restore_tag='prior-'+str(i))
                       for i, row in enumerate(self.proven)]
        self.canonical = [dict(row, pid=old.PIDS[i]) for i, row in enumerate(self.before)]
        vendor = '/private/libGLX_nvidia.so.580.173.02'
        core = '/private/libnvidia-glcore.so.580.173.02'
        self.lines = [f'1000-2000 r-xp 00000000 08:01 {i+1} {path}' for i, path in enumerate((vendor, core))]
        self.inventory = {path: dict(sha256='a'*64, device='08:01', inode=i+1)
                          for i, path in enumerate((vendor, core))}
        self.runtime = dict(schema_version='riley.round14-private-runtime.v1', proven_sessions=copy.deepcopy(self.proven),
                            files={path: 'a'*64 for path in (vendor, core)},
                            helpers={str(self.root/'remote_session_round14.py'): m.PREDECESSOR_SHA,
                                     str(self.root/'remote_session.py'): m.BASE_SHA})
        self.restored = dict(all_relaunched_processes_have_pinned_vendor_maps=True,
                             alive_and_listening=True, commands_and_gui_environment_match=True,
                             host_or_global_configuration_modified=False,
                             processes=[dict(original_pid=row['pid'], new_pid=30+i, start=str(300+i), port=row['port'],
                                             tag='new-'+str(i), runtime_mode='validated_private_compute_and_gl_environment',
                                             driver_runtime=dict(ready=True, all_observed_vendor_mappings_pinned=True,
                                                                 private_vendor_files=copy.deepcopy(self.inventory),
                                                                 driver_mappings=self.lines))
                                        for i, row in enumerate(self.before)])
        self.gate = {'closed_lanes': 2, 'performance_or_acceptance_inferred': False}
        (self.root/'build_decode_profile_batch8.py').write_text(
            'from pathlib import Path\nROOT = Path('+repr(str(self.root))+')\n'
            'def campaign_gate():\n    return '+repr(self.gate)+'\n')
        self.subject = dict(vars(old), CAMPAIGN=self.root, PREVIOUS=self.previous,
                            PREDECESSOR_HELPER_SHA=m.PREDECESSOR_SHA, PREDECESSOR_BASE_SHA=m.BASE_SHA,
                            GENERATION_GATE=self.gate)
        exec(m.CHAIN_SOURCE, self.subject)
        self.calls = []
        def previous_chain(context):
            self.calls.append(copy.deepcopy(context))
            old.ensure(context == self.context, 'original GUI context changed')
            return copy.deepcopy(self.proven)
        self.subject['predecessor_module'] = lambda: SimpleNamespace(qualified_successors=previous_chain)
        self.save()

    def save(self):
        (self.previous/'runtime.json').write_text(json.dumps(self.runtime))
        runtime_ref = {'path': str(self.previous/'runtime.json'), 'sha256': m.sha(self.previous/'runtime.json')}
        self.restored['runtime_manifest'] = runtime_ref
        for row in self.restored['processes']:
            row['runtime_manifest'] = runtime_ref
        for path, value in ((self.previous/'session.json', self.before),
                            (self.previous/'verified.json', self.restored),
                            (self.root/'blender-session.json', self.canonical)):
            path.write_text(json.dumps(value))
        self.subject['PREVIOUS_PINS'] = {name: m.sha(self.previous/name)
                                         for name in ('session.json','runtime.json','verified.json')}
        self.subject['GENERATION_PINS'] = {str(path.relative_to(self.root)): m.sha(path)
                                           for path in self.root.rglob('*.json')}
        self.subject['GENERATION_PINS']['build_decode_profile_batch8.py'] = m.sha(self.root/'build_decode_profile_batch8.py')

    def resolve(self, context=None):
        return self.subject['qualified_successors'](self.context if context is None else context)

    def test_two_generation_chain_selects_latest_not_original_context(self):
        result = self.resolve()
        self.assertEqual(self.calls, [self.context])
        self.assertEqual([row['pid'] for row in result], [30,31,32])
        self.assertEqual([row['port'] for row in result], old.PORTS)
        self.assertEqual([row['gui_env'] for row in result], [row['gui_env'] for row in self.context])

    def test_preserved_original_is_bound_to_its_birth_and_private_maps(self):
        after = self.restored['processes'][0]
        after.update(new_pid=20, start='200', runtime_mode='preserved_original_not_relaunched',
                     driver_runtime=dict(driver_mappings=self.lines, private_runtime_launch_verified=False))
        self.save()
        self.assertEqual(self.resolve()[0]['pid'],20)
        after['start']='999'
        self.save()
        with self.assertRaisesRegex(RuntimeError, 'preserved original identity'): self.resolve()

    def test_rehashed_chain_environment_command_and_port_forgery_rejected(self):
        mutations = [lambda: self.runtime['proven_sessions'][0].update(pid=999),
                     lambda: self.before[0]['env'].update(DISPLAY=':9'),
                     lambda: self.before[0].update(argv=['another-app']),
                     lambda: self.restored['processes'][0].update(port=999),
                     lambda: self.restored['processes'][1].update(new_pid=30),
                     lambda: self.restored['processes'][0].update(original_pid=999),
                     lambda: self.runtime['helpers'].update({str(self.root/'remote_session_round14.py'): '0'*64})]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                snapshot=copy.deepcopy((self.runtime,self.before,self.restored,self.canonical))
                mutation();self.save()
                with self.assertRaises(RuntimeError): self.resolve()
                self.runtime,self.before,self.restored,self.canonical=snapshot
                self.save()
        bad=copy.deepcopy(self.context);bad[0]['gui_env']['DISPLAY']=':9'
        with self.assertRaisesRegex(RuntimeError,'original GUI context'): self.resolve(bad)

    def test_rehashed_vendor_mapping_or_restore_claim_forgery_rejected(self):
        mutations = [lambda r:r.update(alive_and_listening=False),
                     lambda r:r.update(host_or_global_configuration_modified=True),
                     lambda r:r['processes'][0].update(runtime_mode='ambient'),
                     lambda r:r['processes'][0].update(tag=''),
                     lambda r:r['processes'][0]['driver_runtime'].update(ready=False),
                     lambda r:r['processes'][0]['driver_runtime'].update(private_vendor_files={}),
                     lambda r:r['processes'][0]['driver_runtime'].update(driver_mappings=[self.lines[0]]),
                     lambda r:r['processes'][0]['driver_runtime'].update(driver_mappings=[self.lines[0]+' (deleted)']),
                     lambda r:r['processes'][0]['driver_runtime'].update(driver_mappings=[self.lines[0].replace('/private/','/foreign/'),self.lines[1]])]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                snapshot=copy.deepcopy(self.restored);mutation(self.restored);self.save()
                with self.assertRaises(RuntimeError): self.resolve()
                self.restored=snapshot;self.save()

    def test_mutated_receipt_or_new_cleanup_gate_cannot_be_used(self):
        (self.previous/'verified.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError,'generation input changed'): self.resolve()
        self.save()
        self.subject['GENERATION_GATE']={'closed_lanes':3}
        with self.assertRaisesRegex(RuntimeError,'cleanup gate changed'): self.resolve()


class FactoryTests(unittest.TestCase):
    def test_all_live_lifecycle_functions_and_runtime_body_are_unchanged(self):
        source=(HERE/'remote_session_round14.py').read_text()
        generated=m.render(source, {name:'1'*64 for name in ('session.json','runtime.json','verified.json')}, {}, {})
        before,after=m.functions(source),m.functions(generated)
        for name in before.keys()-{'qualified_successors','validate_runtime'}:
            self.assertEqual(before[name],after[name],name)
        normalized=generated.replace("return {'schema_version': 'riley.round15-private-runtime.v1'",
                                     "return {'schema_version': 'riley.round14-private-runtime.v1'",1)
        self.assertEqual(before['validate_runtime'],m.functions(normalized)['validate_runtime'])
        self.assertIn("ROOT = CAMPAIGN / 'blender-round15'",generated)
        self.assertIn("PREVIOUS = CAMPAIGN / 'blender-round14'",generated)
        with self.assertRaisesRegex(ValueError,'unknown predecessor'):
            m.render(source+'\n',{}, {}, {})

    def test_incomplete_campaign_cannot_create_a_helper(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve()
            builder=SimpleNamespace(ROOT=root,campaign_gate=lambda: (_ for _ in ()).throw(FileNotFoundError('finalization.json')))
            with patch.object(m,'CAMPAIGN',root),patch.object(m,'load_pinned',return_value=builder):
                with self.assertRaises(FileNotFoundError):m.prepare(root)
            self.assertFalse((root/'remote_session_round15.py').exists())
            self.assertFalse((root/'round15-session-preparation.json').exists())

    def test_changed_second_gate_prevents_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();(root/'blender-round14').mkdir();(root/'driver-runtime-gui-probe-v2').mkdir()
            (root/'blender-round14/runtime.json').write_text('{}')
            (root/'remote_session_round14.py').write_text('not rendered in this orchestration-only fixture')
            (root/'driver-runtime-gui-probe-v2/completion.json').write_text('{"blender_after": []}')
            pins={'blender-round14/'+name:'1'*64 for name in ('session.json','runtime.json','verified.json')}
            predecessor=SimpleNamespace(bound_runtime=lambda:{})
            base=SimpleNamespace(__file__=str(root/'remote_session.py'))
            with patch.object(m,'CAMPAIGN',root),patch.object(m,'load_pinned',side_effect=[object(),predecessor]),\
                 patch.object(m,'gate_inputs',side_effect=[({'gate':1},pins),({'gate':2},pins)]),\
                 patch.object(m,'render',return_value='def qualified_successors(context): return []\n'),\
                 patch.dict(sys.modules,{'remote_session':base}):
                with self.assertRaisesRegex(ValueError,'changed during preparation'):m.prepare(root)
            self.assertFalse((root/'remote_session_round15.py').exists())

    def test_frozen_cleanup_gate_rejects_missing_exit_and_unbound_preparation(self):
        builder=module('round15_real_gate_fixture',HERE/'build_decode_profile_batch8.py')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);campaign=root/'token-serving-round14';campaign.mkdir()
            def write(path,value):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
            plan=root/'token-serving-round14-plan.json';write(plan,{})
            proof={'path':str(plan),'sha256':m.sha(plan)}
            write(campaign/'preparation.json',{'plan':proof})
            restored=root/'blender-round14/verified.json'
            write(restored,dict(alive_and_listening=True,commands_and_gui_environment_match=True,
                                all_relaunched_processes_have_pinned_vendor_maps=True,
                                host_or_global_configuration_modified=False,processes=[1,2,3]))
            write(campaign/'finalization.json',{'restoration':{'path':str(restored),'sha256':m.sha(restored)}})
            lane=campaign/'pair01/baseline';write(lane/'launch.json',{})
            with patch.object(builder,'ROOT',root),patch.object(builder,'CAMPAIGN_PLAN_SHA',m.sha(plan)):
                with self.assertRaisesRegex(AssertionError,'process exit'):builder.campaign_gate()
                write(lane/'process-exit.json',{'cleanup':{'cleanup_verified':True,'remaining_owned_pids':[]}})
                self.assertEqual(len(builder.campaign_gate()['process_exit_receipts']),1)
                write(campaign/'preparation.json',{'plan':dict(proof,sha256='0'*64)})
                with self.assertRaises(AssertionError):builder.campaign_gate()


if __name__=='__main__':
    unittest.main()
