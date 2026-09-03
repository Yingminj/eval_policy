import numpy as np, glob, subprocess, io
from PIL import Image
R='/mnt/robot_platform/datasets/tidy_up_stationery_le/'
def band(d,n):
    fs=sorted(glob.glob(R+d+'/videos/observation.images.top/chunk-000/*.mp4'))[:n]
    vals=[]
    for f in fs:
        p=subprocess.run(['/usr/bin/ffmpeg','-loglevel','error','-i',f,'-vf','select=not(mod(n\\,200)),scale=320:240,crop=200:35:60:0','-vsync','0','-f','image2pipe','-vcodec','png','-'],capture_output=True)
        for pt in p.stdout.split(b'\x89PNG\r\n\x1a\n')[1:]:
            im=np.asarray(Image.open(io.BytesIO(b'\x89PNG\r\n\x1a\n'+pt)).convert('L'),dtype=np.float32)
            vals.append(im.mean())
    return np.array(vals)
a=band('batch_success_361',16); b=band('batch_fail_72',16)
print('top-band brightness  succ: mean %.1f std %.1f  n=%d'%(a.mean(),a.std(),len(a)))
print('top-band brightness  fail: mean %.1f std %.1f  n=%d'%(b.mean(),b.std(),len(b)))
thr=(a.mean()+b.mean())/2
acc=max(((a>thr).mean()+(b<thr).mean())/2, ((a<thr).mean()+(b>thr).mean())/2)
print('1-feature (mean brightness of background band) separation accuracy: %.3f'%acc)
