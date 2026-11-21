"""Whole-motion split supervised 4D target/time -> 64D SONIC-token prior."""
import argparse, copy, subprocess, time
import torch
from torch import nn
from common import *
class Prior(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,64))
    def forward(self,x):return self.net(x)

def build(output):
    output.mkdir(parents=True,exist_ok=True);encoder=MotionEncoder();ms=motions()
    xs=[];ys=[];ids=[];provenance=[]
    for i,m in enumerate(ms):
        for t in m['time']:
            q,dq=future(m,float(t));xs.append(np.r_[m['target'],1-t]);ys.append(encoder.encode(q,dq,np.eye(3)));ids.append(i)
        provenance.append({'motion':m['name'],'npz_sha256':sha(GRID/m['name']/'reference.npz')})
        print('encoded',i+1,len(ms),m['name'],flush=True)
    # Interior target combinations held out as entire clips; no frame leakage.
    validation=[i for i,m in enumerate(ms) if m['name'] in ('table_tennis_forehand_y-005_z005','table_tennis_forehand_y-010_z010')]
    test=[i for i,m in enumerate(ms) if m['name'] in ('table_tennis_forehand_y-010_z005','table_tennis_forehand_y-005_z010')]
    if len(validation)!=2 or len(test)!=2:raise ValueError('Expected held-out grid cells absent')
    train=[i for i in range(len(ms)) if i not in validation+test]
    np.savez_compressed(output/'paired.npz',x=np.array(xs,dtype=np.float32),token=np.array(ys,dtype=np.float32),motion_id=np.array(ids),
        names=np.array([m['name'] for m in ms]),train_ids=train,validation_ids=validation,test_ids=test)
    meta={'source_url':'https://github.com/dvij542/table-tennis-motion-kit','source_commit':subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip(),
        'motions':provenance,'encoder_sha256':sha(DECODER.parent/'model_encoder.onnx'),'decoder_sha256':sha(DECODER),
        'features':['target_x_pelvis_m','target_y_pelvis_m','target_z_pelvis_m','time_before_hit_s'],
        'output':'64D SONIC token','sampling_hz':50,'future_window_s':np.arange(10).tolist(),
        'future_sample_step_s':.1,'contact_offset_wrist_m':CONTACT.tolist(),'reference_mount_not_hardware_calibration':True,
        'split':{'train':train,'validation':validation,'test':test},'simulation_only':True,'hardware_approved':False,
        'notes':['Reference root orientation is identity when generating labels. Frozen decoder receives current proprioception.',
                 'Only 20 target trajectories, X fixed at 0.43 m. No claim of target extrapolation.',
                 'Time before hit includes follow-through/recovery: 1.0 to -1.5 s.']}
    meta['future_window_s']=(np.arange(10)*.1).tolist()
    (output/'dataset.json').write_text(json.dumps(meta,indent=2))

def train(output,epochs):
    torch.set_num_threads(2);torch.manual_seed(501)
    a=np.load(output/'paired.npz',allow_pickle=False);meta=json.loads((output/'dataset.json').read_text())
    masks={name:np.isin(a['motion_id'],a[f'{name}_ids']) for name in ('train','validation','test')}
    xm=a['x'][masks['train']].mean(0);xs=np.maximum(a['x'][masks['train']].std(0),.01)
    ym=a['token'][masks['train']].mean(0);ys=np.maximum(a['token'][masks['train']].std(0),.03)
    x=torch.tensor((a['x']-xm)/xs);y=torch.tensor((a['token']-ym)/ys)
    model=Prior();opt=torch.optim.Adam(model.parameters(),lr=.002)
    schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.00003)
    tr=masks['train'];va=masks['validation'];best=float('inf');history=[]
    for epoch in range(epochs):
        model.train();pred=model(x[tr]);truth=y[tr]
        p=pred.reshape(-1,126,64);q=truth.reshape(-1,126,64)
        loss=(pred-truth).square().mean()+.1*((p[:,1:]-p[:,:-1])-(q[:,1:]-q[:,:-1])).square().mean()
        opt.zero_grad();loss.backward();opt.step();schedule.step()
        if epoch%20==0 or epoch==epochs-1:
            model.eval()
            with torch.no_grad():vl=float((model(x[va])-y[va]).square().mean())
            if vl<best:best=vl;weights=copy.deepcopy(model.state_dict());best_epoch=epoch
            history.append({'epoch':epoch,'train_loss':float(loss),'validation_loss':vl})
            if epoch%200==0:print(history[-1],flush=True)
    model.load_state_dict(weights);model.eval();metrics={}
    with torch.no_grad():pred=model(x).numpy()*ys+ym
    for name,mask in masks.items():
        e=pred[mask]-a['token'][mask];metrics[name]={'motions':int(len(a[f'{name}_ids'])),'frames':int(mask.sum()),'token_rmse':float(np.sqrt(np.mean(e*e))),
            'normalized_rmse':float(np.sqrt(np.mean((e/ys)**2))),'max_absolute_token_error':float(abs(e).max())}
    cp={'state_dict':weights,'x_mean':xm.tolist(),'x_scale':xs.tolist(),'y_mean':ym.tolist(),'y_scale':ys.tolist(),
        'simulation_only':True,'hardware_approved':False,'decoder_sha256':meta['decoder_sha256'],'encoder_sha256':meta['encoder_sha256'],
        'dataset_sha256':sha(output/'paired.npz'),'architecture':'4-128-128-128-64 tanh','best_epoch':best_epoch}
    torch.save(cp,output/'policy.pt');(output/'training.json').write_text(json.dumps({'metrics':metrics,'best_epoch':best_epoch,'history':history},indent=2))
    print(json.dumps(metrics,indent=2),flush=True)

def load(path):
    cp=torch.load(path,map_location='cpu',weights_only=True)
    if (not cp['simulation_only'] or cp['hardware_approved'] or cp['decoder_sha256']!=sha(DECODER)
            or cp['encoder_sha256']!=sha(DECODER.parent/'model_encoder.onnx')):raise ValueError('Incompatible checkpoint')
    model=Prior();model.load_state_dict(cp['state_dict']);model.eval()
    def predict(x):
        x=np.asarray(x,dtype=np.float32)
        if x.shape!=(4,) or not np.isfinite(x).all():raise ValueError('Need four finite target/time features')
        # Reject extrapolation; target uses pelvis frame, not world coordinates.
        if abs(x[0]-.43)>.002 or not -.531<=x[1]<=-.329 or not .077<=x[2]<=.229 or not -1.501<=x[3]<=1.001:raise ValueError('Outside training target/time domain')
        with torch.no_grad():return model(torch.tensor((x-np.array(cp['x_mean']))/np.array(cp['x_scale']),dtype=torch.float32)).numpy()*np.array(cp['y_scale'])+np.array(cp['y_mean'])
    return predict

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=RUN);p.add_argument('--epochs',type=int,default=2400);p.add_argument('--reuse-data',action='store_true');args=p.parse_args()
    if not args.reuse_data:build(args.output)
    train(args.output,args.epochs)
