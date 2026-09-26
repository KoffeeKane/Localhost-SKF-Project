"""Required no-PPO gate for the full v4 training run."""
import hashlib
import json
from pathlib import Path
import numpy as np
from g1_stair_env import G1StairEnv
from gait_metrics import GaitMetrics

p=Path(__file__).parent
env=G1StairEnv(mode='flat')
results=[]
for episode in range(3):
    _,info=env.reset()
    metrics=GaitMetrics(info)
    max_x=info['x']
    trace=[info]
    for _ in range(1000):
        obs,r,t,tr,info=env.step(np.zeros(29))
        assert np.isfinite(obs).all() and np.isfinite(r)
        assert np.all(env.data.ctrl>=env.torque_limits[:,0]) and np.all(env.data.ctrl<=env.torque_limits[:,1])
        metrics.update(info); trace.append(info); max_x=max(max_x,info['x'])
        if t or tr:break
    summary=metrics.summary()
    summary.update(episode=episode+1,steps=info['episode_step'],max_x=max_x,final_x=info['x'],fallen=info['fallen'])
    results.append(summary)
    (p/'logs/v4_reference'/f'validated_episode_{episode+1}.json').write_text(json.dumps(trace,indent=2))
passed=all(r['valid_left_steps']>=1 and r['valid_right_steps']>=1 and r['alternating_step_pairs']>=1
           and all(r['real_unloading'].values()) for r in results)
hashes={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ['g1_stair_env.py','g1_gait_env_v3.py','g1_pd_env_v2.py','gait_metrics.py']}
report=dict(passed=passed,environment_hashes=hashes,episodes=results,
            gate='At least one valid left and right step and an alternating pair, with measured unloading; zero residual.')
(p/'logs/v4_reference/validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
if not passed:raise RuntimeError('Reference-only gate failed; do not start full training')
