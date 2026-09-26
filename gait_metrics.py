"""Contact-force-supported lift, advance, and replant event detector."""
import numpy as np


class GaitMetrics:
    def __init__(self,info):
        self.events=[]
        self.lifts=[]
        self.state={s:dict(base=0.0,x=info.get(f'{s}_foot_x',0),lifted=False,forward=False,land=0) for s in ['left','right']}
        self.max_clearance=dict(left=0.0,right=0.0)
        self.max_height=dict(left=0.0,right=0.0)
        self.count=0
        self.slip_sum=0.0
        self.saturation=0.0
        self.crouch_count=0
        self.reward_sums={}
        self.unloaded=dict(left=False,right=False)

    def update(self,i):
        if 'left_contact_force' not in i: return
        self.count+=1
        bw=i['body_weight']
        self.saturation+=i['residual_saturation_percentage']
        self.slip_sum+=i[f"{i['stance_leg']}_foot_slip_speed"]
        self.crouch_count+=int(i['left_foot_contact'] and i['right_foot_contact'] and min(i['left_knee_angle'],i['right_knee_angle'])>.55)
        for side,other in [('left','right'),('right','left')]:
            s=self.state[side]
            h,x,force=i[f'{side}_foot_height'],i[f'{side}_foot_x'],i[f'{side}_contact_force']
            self.max_height[side]=max(self.max_height[side],h)
            support=i[f'{other}_contact_force']>.55*bw
            low=force<.15*bw
            upright=i['upright']>.85 and i['torso_upright']>.85
            if i['swing_leg']==side and support and low and upright:
                self.unloaded[side]=True
                if h-s['base']>.02 and not i[f'{side}_foot_contact'] and not s['lifted']:
                    s['lifted']=True
                    self.lifts.append(dict(leg=side,step=i['episode_step']))
            if s['lifted']:
                self.max_clearance[side]=max(self.max_clearance[side],h-s['base'])
                s['forward'] |= x-s['x']>.02
                planted=force>.15*bw and i[f'{side}_foot_contact'] and h-s['base']<.008
                s['land']=s['land']+1 if planted else 0
                if s['land']>=3:
                    if s['forward'] and upright:
                        self.events.append(dict(leg=side,step=i['episode_step'],advance=x-s['x']))
                    s.update(lifted=False,forward=False,land=0,base=h,x=x)
            elif force>.15*bw and i[f'{side}_foot_contact']:
                s.update(base=h,x=x)
            self.max_clearance[side]=max(self.max_clearance[side],h-s['base'])
        for k,v in i.get('reward_terms',{}).items():
            self.reward_sums[k]=self.reward_sums.get(k,0)+float(v)

    def summary(self):
        legs=[e['leg'] for e in self.events]
        lifted=[e['leg'] for e in self.lifts]
        pairs=sum(a!=b for a,b in zip(legs,legs[1:]))
        return dict(valid_left_steps=legs.count('left'),valid_right_steps=legs.count('right'),
                    alternating_step_pairs=pairs,valid_step_events=self.events,qualified_lift_events=self.lifts,
                    alternating_foot_lift=any(a!=b for a,b in zip(lifted,lifted[1:])),
                    real_unloading=self.unloaded, max_left_foot_clearance=self.max_clearance['left'],
                    max_right_foot_clearance=self.max_clearance['right'],
                    max_left_foot_height=self.max_height['left'],max_right_foot_height=self.max_height['right'],
                    average_stance_foot_slip=self.slip_sum/max(1,self.count),
                    crouch_fraction=self.crouch_count/max(1,self.count),
                    mean_residual_saturation_percentage=self.saturation/max(1,self.count),
                    mean_reward_terms={k:v/max(1,self.count) for k,v in self.reward_sums.items()})
