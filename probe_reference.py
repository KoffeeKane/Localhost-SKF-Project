import argparse
import json
from pathlib import Path
import numpy as np
from g1_stair_env import G1StairEnv

p=argparse.ArgumentParser()
p.add_argument('--config',default='{}')
p.add_argument('--output',type=Path,required=True)
p.add_argument('--steps',type=int,default=1000)
a=p.parse_args()
e=G1StairEnv(mode='flat',reference_config=json.loads(a.config))
o,i=e.reset()
trace=[i]
for n in range(a.steps):
    o,r,t,tr,i=e.step(np.zeros(29))
    trace.append(i)
    if t or tr: break
summary=dict(steps=i['episode_step'],x=i['x'],fallen=i['fallen'],max_clearance=[max(v[f'{s}_foot_height'] for v in trace) for s in ['left','right']],
             min_forces=[min(v[f'{s}_contact_force'] for v in trace[10:]) for s in ['left','right']])
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(config=e.cfg,summary=summary,trace=trace),indent=2))
print(json.dumps(summary))
for v in trace[::25]:
    print(json.dumps({k:v[k] for k in ['episode_step','phase','subphase','lateral_reference','com_y','com_stance_lateral_offset','left_contact_force','right_contact_force','left_foot_height','right_foot_height','upright']}))
