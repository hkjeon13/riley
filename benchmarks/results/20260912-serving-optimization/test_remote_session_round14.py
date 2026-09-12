import ast
import copy
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('round14_test_subject', HERE/'remote_session_round14.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Tests(unittest.TestCase):
    def setUp(self):
        self.context = [dict(pid=10+i, start=str(100+i), port=p, gui_env={'DISPLAY': ':1'}) for i,p in enumerate(m.PORTS)]
        self.before = [dict(pid=x['pid'], start=x['start'], port=x['port'], env=x['gui_env']) for x in self.context]
        runtime_ref = {'path': str(m.PREVIOUS/'runtime.json'), 'sha256': m.PREVIOUS_PINS['runtime.json']}
        self.receipt = dict(runtime_manifest=runtime_ref,
            all_relaunched_processes_have_pinned_vendor_maps=True, alive_and_listening=True,
            commands_and_gui_environment_match=True, host_or_global_configuration_modified=False,
            processes=[dict(original_pid=x['pid'],new_pid=20+i,start=str(200+i),port=x['port'],
                runtime_manifest=runtime_ref,runtime_mode='validated_private_compute_and_gl_environment',
                driver_runtime=dict(ready=True,all_observed_vendor_mappings_pinned=True,private_vendor_files={'library': 'proof'}))
                for i,x in enumerate(self.context)])
        self.docs = {'session.json': self.before, 'verified.json': self.receipt, 'runtime.json': {'proven_sessions': self.context}}
        for patcher in [patch.object(m,'digest',lambda p:m.PREVIOUS_PINS[p.name]),
                        patch.object(m,'read',lambda p:copy.deepcopy(self.docs[p.name]))]:
            patcher.start();self.addCleanup(patcher.stop)

    def test_valid_three_successors(self):
        rows=m.qualified_successors(self.context)
        self.assertEqual([x['pid'] for x in rows],[20,21,22])
        self.assertEqual([x['port'] for x in rows],m.PORTS)
        self.assertEqual([x['gui_env'] for x in rows],[x['gui_env'] for x in self.context])

    def test_reject_changed_receipt(self):
        with patch.object(m,'digest',return_value='0'*64),self.assertRaisesRegex(RuntimeError,'predecessor receipt changed'):
            m.qualified_successors(self.context)

    def test_reject_scope_or_runtime_changes(self):
        mutations=[lambda r:r['processes'][0].update(original_pid=999),
                   lambda r:r['processes'][1].update(new_pid=20),
                   lambda r:r.update(host_or_global_configuration_modified=True),
                   lambda r:r['processes'][0]['driver_runtime'].update(ready=False),
                   lambda r:r['processes'][0].update(runtime_mode='ambient'),
                   lambda r:r['processes'][0].update(port=9999)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                original=copy.deepcopy(self.receipt);mutate(self.receipt)
                with self.assertRaises(RuntimeError):m.qualified_successors(self.context)
                self.receipt.clear();self.receipt.update(original)

    def test_reject_context_environment_change(self):
        changed=copy.deepcopy(self.context);changed[0]['gui_env']['DISPLAY']=':9'
        with self.assertRaisesRegex(RuntimeError,'GUI context provenance'):m.qualified_successors(changed)

    def test_lifecycle_unchanged_from_qualified_round13(self):
        previous=(HERE/'remote_session_round13_v2.py').read_text().replace('round13','round14').replace('round12','round13')
        current=(HERE/'remote_session_round14.py').read_text()
        def functions(text):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
        old,new=functions(previous),functions(current)
        for name in old:
            if name not in {'validate_runtime','check'}:
                self.assertEqual(old[name],new[name],name)


if __name__=='__main__':
    unittest.main()
