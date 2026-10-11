"""Real Windows venv-parent and competing-process admission probes; no run."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from tools import run_five_baselines_amem_native_v4 as v4

@unittest.skipUnless(os.name=='nt','Windows process gate')
class V4RealProcessGateTests(unittest.TestCase):
    def test_real_venv_launcher_chain_passes_but_other_controller_blocks(self):
        python=v4.ROOT/'runtime/baseline_env_v1/Scripts/python.exe'
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            own=root/'run_five_baselines_amem_native_v4.py'
            own.write_text('import sys\nsys.path.insert(0,'+repr(str(v4.ROOT))+')\nfrom tools import run_five_baselines_amem_native_v4 as v4\nv4.assert_source_stopped()\nprint("GATE_PASSED_NO_RUN",flush=True)\n',encoding='utf-8')
            first=subprocess.run([str(python),str(own),'run'],capture_output=True,text=True,timeout=30,creationflags=0x08000000)
            self.assertEqual(first.returncode,0,first.stderr)
            self.assertEqual(first.stdout.strip(),'GATE_PASSED_NO_RUN')
            blocker=root/'run_five_baselines_other.py'
            blocker.write_text('import sys\nprint("BLOCKER_READY_NO_RUN",flush=True)\nsys.stdin.read()\n',encoding='utf-8')
            process=subprocess.Popen([str(python),str(blocker),'run'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=0x08000000)
            try:
                self.assertEqual(process.stdout.readline().strip(),'BLOCKER_READY_NO_RUN')
                second=subprocess.run([str(python),str(own),'run'],capture_output=True,text=True,timeout=30,creationflags=0x08000000)
                self.assertNotEqual(second.returncode,0)
                self.assertIn('baseline controller is active',second.stderr)
            finally:
                process.communicate('',timeout=20)
            self.assertEqual(process.returncode,0)

if __name__=='__main__':unittest.main()
