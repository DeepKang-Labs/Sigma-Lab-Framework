"""Run the verified local diagnostic route without downloading models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

root = Path(__file__).resolve().parents[1]
run = root/'.tmp/local-checks'/uuid.uuid4().hex
run.mkdir(parents=True)
env = dict(os.environ,PYTHONPATH=str(root))
steps = [('tests',['-m','pytest','-q','-rs'])]
for network in ['skywire','fiber']:
    steps.append((network,['-m','network_bridge.run_network_integrated','--network',network,
        '--discovery',str(run),'--mappings',str(root/f'network_bridge/mappings_{network}.yaml'),
        '--config',str(root/'sigma_config_placeholder.yaml'),'--out',str(run/f'{network}.json'),
        '--validate-only','--formula-eval','auto']))
steps += [('memory',['-m','tools.mesh_memory_append','--report',str(run/'skywire.json'),'--memory',str(run/'memory.json')]),
          ('priority',['-m','tools.priority_matrix_from_mappings','--mappings',str(root/'network_bridge/mappings_skywire.yaml'),
                       '--out',str(run/'priority.json')])]
results=[]
print('Local check directory: '+str(run),flush=True)
for name,command in steps:
    process=subprocess.run([sys.executable,*command],cwd=root,env=env,capture_output=True,text=True,timeout=120)
    (run/(name+'.log')).write_text(process.stdout+'\n'+process.stderr,encoding='utf-8')
    results.append({'step':name,'exit_code':process.returncode})
    print(name+': '+('OK' if process.returncode==0 else 'FAILED'),flush=True)
    if process.returncode:
        print(process.stderr)
        break
summary={'scope':'local-diagnostics-only','live_transport_established':False,'steps':results,
         'model_integration_executed':False,'status':'completed' if len(results)==len(steps) and all(r['exit_code']==0 for r in results) else 'failed'}
(run/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
sys.exit(0 if summary['status']=='completed' else 1)
