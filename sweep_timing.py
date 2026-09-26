import json
from pathlib import Path
import numpy as np
from g1_stair_env import G1StairEnv
configs=[dict(period=2.0,shift=.07,com_damping=2.0),dict(period=2.4,shift=.06,com_damping=2.0),dict(period=3.0,shift=.05,com_damping=2.0),dict(period=3.0,shift=.07,com_gain=-2.0,com_damping=-.6)]
for n,c in enumerate(configs):
 e=G1StairEnv(mode='flat',reference_config=c); _,i=e.reset(); trace=[i]
 for k in range(1000):
  _,_,t,tr,i=e.step(np.zeros(29)); trace.append(i)
  if t or tr: break
 good={s:sum(v[f'{s}_foot_height']>.015 and v['upright']>.9 and v[f'{s}_contact_force']<20 and v[f'{o}_contact_force']>200 for v in trace) for s,o in [('left','right'),('right','left')]}
 print(n,c,'steps',i['episode_step'],'x',i['x'],'good',good,flush=True)
 Path(f'logs/v4_reference/timing_{n}.json').write_text(json.dumps(dict(config=e.cfg,trace=trace)))

