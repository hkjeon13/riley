import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('remote_session_round3.py')
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location('round2', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
real_tagged_process = m.tagged_process
real_process_tag = m.process_tag

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='riley-round3-mock-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.rows = [dict(pid=10+i, start=str(100+i), argv=['/bin/blender', '--port', str(port)],
                          cwd='/original/gui', env={'DISPLAY': ':0'}, port=port)
                     for i, port in enumerate(m.PORTS)]
        self.proc = {}
        self.launched = []
        self.clock = 0
        self.fail_identity = False
        self.patchers = []
        for name, value in [('ROOT', self.root), ('SNAPSHOT', self.root/'session.json'),
                            ('require_pidfds', lambda: None), ('process_start', self.start),
                            ('live', self.live), ('identity', self.ident),
                            ('process_tag', lambda pid: self.proc[pid].get('tag')),
                            ('tagged_process', self.discover), ('listening', self.listening)]:
            self.patchers.append(patch.object(m, name, value))
        self.patchers += [patch.object(m.subprocess, 'Popen', self.popen),
                          patch.object(m.time, 'sleep', lambda sec: None),
                          patch.object(m.time, 'monotonic', self.tick)]
        for p in self.patchers: p.start(); self.addCleanup(p.stop)
        m.write(m.SNAPSHOT, self.rows)
        m.write(self.root/'stop-intents.json', [r['pid'] for r in self.rows])
        self.output = contextlib.ExitStack()
        self.output.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.output.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.addCleanup(self.output.close)

    def tick(self): self.clock += 31; return self.clock
    def start(self, pid): return self.proc.get(pid, {}).get('start')
    def live(self, pid): return self.proc.get(pid, {}).get('live', False)
    def ident(self, pid):
        if self.fail_identity and pid >= 200:
            self.fail_identity = False
            raise RuntimeError('crash after Popen')
        p = self.proc[pid]
        return {k: copy.deepcopy(p[k]) for k in ('pid', 'start', 'argv', 'cwd', 'env')}
    def discover(self, row, tag):
        found = [self.ident(pid) for pid,p in self.proc.items() if p.get('tag') == tag and p['live']]
        m.ensure(len(found) <= 1, 'multiple tags')
        for p in found: m.ensure(m.same_command(p,row), 'command differs')
        return found[0] if found else None
    def listening(self, port):
        return any(p['live'] and p['port'] == port for p in self.proc.values())
    def popen(self, argv, **kw):
        row = next(r for r in self.rows if r['argv'] == argv)
        saved = json.loads((self.root/'restoring.json').read_text())
        item = next(x for x in saved if x['original_pid'] == row['pid'])
        assert item['launch_intent'] and item['tag'] == kw['env'][m.TAG_KEY]
        assert kw['cwd'] == row['cwd'] and kw['start_new_session']
        assert {k:v for k,v in kw['env'].items() if k in m.GUI_KEYS} == row['env']
        pid = 200 + len(self.launched)
        self.launched.append(pid)
        self.proc[pid] = dict(row, pid=pid, start=str(pid*10), live=True, tag=item['tag'])
        return types.SimpleNamespace(pid=pid)
    def journal(self, original, pid=None, start=None, tag='unique-tag'):
        item = dict(original_pid=original['pid'], port=original['port'], tag=tag, launch_intent=True)
        if pid is not None: item.update(new_pid=pid,start=start)
        m.write(self.root/'restoring.json', [item])
    def assert_verified(self):
        receipt=json.loads((self.root/'verified.json').read_text())
        self.assertTrue(receipt['alive_and_listening'])
        self.assertEqual(len(receipt['processes']),3)
        for file in self.root.iterdir(): self.assertEqual(file.stat().st_mode & 0o777,0o600)

    def test_normal_restore_exact_env_private_state_and_three_launches(self):
        with patch.dict(os.environ, {'WAYLAND_DISPLAY':'bad','XAUTHORITY':'bad'}): m.restore()
        self.assertEqual(len(self.launched),3); self.assert_verified()
        m.restore(); self.assertEqual(len(self.launched),3)

    def test_crash_after_launch_is_adopted_without_duplicate(self):
        self.fail_identity=True
        with self.assertRaisesRegex(RuntimeError,'crash after Popen'): m.restore()
        self.assertEqual(self.launched,[200])
        m.restore(); self.assertEqual(len(self.launched),3); self.assert_verified()

    def test_dead_same_start_successor_replaced(self):
        self.proc[190]=dict(self.rows[0],pid=190,start='old',live=False,tag='unique-tag')
        self.journal(self.rows[0],190,'old')
        m.restore(); self.assertEqual(len(self.launched),3); self.assert_verified()

    def test_absent_journaled_successor_adopts_unjournaled_replacement(self):
        self.journal(self.rows[0],190,'old')
        self.proc[199]=dict(self.rows[0],pid=199,start='new',live=True,tag='unique-tag')
        m.restore(); self.assertEqual(len(self.launched),2); self.assert_verified()

    def test_reused_journaled_successor_fails_closed(self):
        self.journal(self.rows[0],190,'old')
        self.proc[190]=dict(self.rows[0],pid=190,start='other',live=True)
        with self.assertRaisesRegex(RuntimeError,'PID was reused'): m.restore()
        self.assertFalse(self.launched); self.assertFalse((self.root/'verified.json').exists())

    def test_reused_original_fails_without_signal(self):
        self.proc[10]=dict(self.rows[0],start='reused',live=True)
        with patch.object(m.signal,'pidfd_send_signal',create=True) as sig:
            with self.assertRaisesRegex(RuntimeError,'original PID was reused'): m.restore()
            sig.assert_not_called()
        self.assertFalse(self.launched)

    def test_terminating_original_never_verified(self):
        self.proc[10]=dict(self.rows[0],live=True)
        with patch.object(m.os,'pidfd_open',lambda pid: os.open(os.devnull,os.O_RDONLY),create=True), \
             patch.object(m.signal,'pidfd_send_signal',create=True) as sig:
            with self.assertRaisesRegex(RuntimeError,'still terminating'): m.restore()
            self.assertEqual(sig.call_count,1)
        self.assertFalse(self.launched); self.assertFalse((self.root/'verified.json').exists())

    def test_partial_stop_original_without_intent_is_preserved(self):
        self.proc[12]=dict(self.rows[2],live=True)
        m.write(self.root/'stop-intents.json',[10,11])
        m.restore(); self.assertEqual(len(self.launched),2); self.assert_verified()
        receipt=json.loads((self.root/'verified.json').read_text())
        self.assertEqual(receipt['processes'][2]['new_pid'],12)

    def test_live_port_without_matching_tag_blocks_launch(self):
        self.proc[199]=dict(self.rows[0],pid=199,start='new',live=True,tag='other')
        with self.assertRaisesRegex(RuntimeError,'port occupied'): m.restore()
        self.assertFalse(self.launched)

    def test_watchdog_retries_restore_exception(self):
        m.write(self.root/'deadline.json',0)
        calls=[]
        def attempt():
            calls.append(1)
            if len(calls)==1: raise RuntimeError('temporary')
            m.write(self.root/'verified.json',{'ok':True})
        with patch.object(m,'restore',attempt),patch.object(m.time,'sleep') as sleep:
            m.watchdog()
            self.assertEqual(len(calls),2)
            self.assertTrue(all(c.args==(5,) for c in sleep.call_args_list))

    def test_direct_tag_discovery_and_duplicate_rejection(self):
        proc_root=self.root/'proc'
        proc_root.mkdir()
        row=copy.deepcopy(self.rows[0])
        row['cwd']=str(self.root.resolve())
        tag='direct-scan-tag'
        def mapped_path(value):
            return proc_root if value == '/proc' else Path(value)
        for pid in (300,301):
            entry=proc_root/str(pid); entry.mkdir()
            (entry/'environ').write_bytes((m.TAG_KEY+'='+tag+'\0').encode())
            (entry/'cmdline').write_bytes(('\0'.join(row['argv'])+'\0').encode())
            (entry/'cwd').symlink_to(row['cwd'])
            self.proc[pid]=dict(row,pid=pid,start=str(pid),live=(pid==300),tag=tag)
        with patch.object(m,'Path',mapped_path),patch.object(m,'process_tag',real_process_tag):
            found=real_tagged_process(row,tag)
            self.assertEqual(found['pid'],300)
            self.assertIsNone(real_tagged_process(row,'unused'))
            self.proc[301]['live']=True
            with self.assertRaisesRegex(RuntimeError,'multiple live processes'):
                real_tagged_process(row,tag)
            self.proc[301]['live']=False
            self.proc[300]['argv']=['/bin/other']
            with self.assertRaisesRegex(RuntimeError,'command or GUI'):
                real_tagged_process(row,tag)

    def test_tag_scan_filters_unrelated_before_environment_and_fails_closed_for_candidate(self):
        proc_root=self.root/'proc-filter'; proc_root.mkdir()
        row=copy.deepcopy(self.rows[0]); row['cwd']=str(self.root.resolve())
        other_cwd=self.root/'other'; other_cwd.mkdir()
        for pid,argv,cwd in [(400,['/bin/unrelated'],row['cwd']),
                             (401,row['argv'],str(other_cwd.resolve())),
                             (402,row['argv'],row['cwd'])]:
            entry=proc_root/str(pid); entry.mkdir()
            (entry/'cmdline').write_bytes(('\0'.join(argv)+'\0').encode())
            (entry/'cwd').symlink_to(cwd)
            self.proc[pid]=dict(row,pid=pid,start=str(pid),live=True)
        touched=[]
        def inaccessible(pid):
            touched.append(pid)
            raise PermissionError('not dumpable')
        def mapped_path(value):
            return proc_root if value == '/proc' else Path(value)
        with patch.object(m,'Path',mapped_path),patch.object(m,'process_tag',inaccessible):
            with self.assertRaises(PermissionError): real_tagged_process(row,'tag')
        self.assertEqual(touched,[402])

    def test_exit_poll_never_reads_environment_after_signal(self):
        fresh=self.root/'exit-round'
        self.proc={r['pid']:dict(r,live=True) for r in self.rows}
        fds={}; signaled=set()
        def open_pidfd(pid):
            fd=os.open(os.devnull,os.O_RDONLY); fds[fd]=pid; return fd
        def sig(fd,signal): signaled.add(fds[fd])
        def identity_before_signal(pid):
            if pid in signaled: raise PermissionError('dying process environ')
            return self.ident(pid)
        def reap(_):
            for pid in signaled: self.proc[pid]['live']=False
        with patch.object(m,'ROOT',fresh),patch.object(m,'SNAPSHOT',fresh/'session.json'), \
             patch.object(m,'check',lambda:copy.deepcopy(self.rows)), \
             patch.object(m,'identity',identity_before_signal), \
             patch.object(m.os,'pidfd_open',open_pidfd,create=True), \
             patch.object(m.signal,'pidfd_send_signal',sig,create=True), \
             patch.object(m.subprocess,'Popen',lambda *a,**kw:None), \
             patch.object(m.time,'monotonic',lambda:0),patch.object(m.time,'sleep',reap):
            m.stop()
        self.assertEqual(signaled,{10,11,12})
        self.assertTrue((fresh/'stopped.json').exists())
        self.assertFalse((fresh/'verified.json').exists())

    def test_full_identity_permission_only_tolerated_after_confirmed_exit(self):
        row=self.rows[0]; self.proc[row['pid']]=dict(row,live=True)
        with patch.object(m,'identity',side_effect=PermissionError('not dumpable')):
            self.assertTrue(m.original_running(row))
            with self.assertRaises(PermissionError): m.original_alive(row)
        def dying(pid):
            self.proc[pid]['live']=False
            raise PermissionError('dying process environ')
        with patch.object(m,'identity',dying): self.assertFalse(m.original_alive(row))

    def test_round2_receipt_chain_requires_canonical_command_birth_and_tag(self):
        campaign=self.root/'campaign'; campaign.mkdir()
        previous=campaign/'blender-round2'; previous.mkdir()
        canonical=[dict(row,pid=pid) for row,pid in zip(self.rows,m.PIDS)]
        prior=[dict(row,pid=210+i) for i,row in enumerate(self.rows)]
        receipt={'alive_and_listening':True,'commands_and_gui_environment_match':True,
                 'processes':[dict(original_pid=old['pid'],new_pid=row['pid'],start=row['start'],
                                   port=row['port'],tag='prior-tag-'+str(i))
                              for i,(row,old) in enumerate(zip(self.rows,prior))]}
        m.write(campaign/'blender-session.json',canonical)
        m.write(previous/'session.json',prior)
        m.write(previous/'verified.json',receipt)
        self.proc={row['pid']:dict(row,live=True,tag='prior-tag-'+str(i))
                   for i,row in enumerate(self.rows)}
        with patch.object(m,'CAMPAIGN',campaign),patch.object(m,'PREVIOUS',previous), \
             patch.object(m.os,'pidfd_open',lambda pid:os.open(os.devnull,os.O_RDONLY),create=True):
            checked=m.check()
            self.assertEqual([r['restore_tag'] for r in checked],['prior-tag-0','prior-tag-1','prior-tag-2'])
            self.proc[10]['tag']='wrong'
            with self.assertRaisesRegex(RuntimeError,'restore tag changed'): m.check()
            self.proc[10]['tag']='prior-tag-0'; self.proc[10]['start']='reused'
            with self.assertRaisesRegex(RuntimeError,'PID was reused'): m.check()
            self.proc[10]['start']=self.rows[0]['start']
            prior[0]['env']={'DISPLAY':':wrong'}; m.write(previous/'session.json',prior)
            with self.assertRaisesRegex(RuntimeError,'canonical process command'): m.check()

    def test_stop_persists_intent_before_exact_pidfd_signal(self):
        # Use a new campaign directory, as stop deliberately refuses an existing round.
        fresh=self.root/'round'
        self.proc={r['pid']:dict(r,live=True) for r in self.rows}
        fds={}; signaled=[]; watchdog_started=[]
        def open_pidfd(pid):
            fd=os.open(os.devnull,os.O_RDONLY); fds[fd]=pid; return fd
        def sig(fd, signal):
            pid=fds[fd]
            self.assertTrue(watchdog_started)
            self.assertIn(pid,json.loads((fresh/'stop-intents.json').read_text()))
            self.assertEqual(signal,m.signal.SIGTERM)
            signaled.append(pid); self.proc[pid]['live']=False
        with patch.object(m,'ROOT',fresh),patch.object(m,'SNAPSHOT',fresh/'session.json'), \
             patch.object(m,'check',lambda:copy.deepcopy(self.rows)), \
             patch.object(m.os,'pidfd_open',open_pidfd,create=True), \
             patch.object(m.signal,'pidfd_send_signal',sig,create=True), \
             patch.object(m.subprocess,'Popen',lambda *a,**kw:watchdog_started.append(True)):
            m.stop()
        self.assertEqual(signaled,[10,11,12])
        self.assertEqual(fresh.stat().st_mode & 0o777,0o700)
        self.assertTrue((fresh/'stopped.json').exists())

if __name__ == '__main__': unittest.main(verbosity=2)
