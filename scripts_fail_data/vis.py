import numpy as np, glob, subprocess, io
from PIL import Image
R='/mnt/robot_platform/datasets/tidy_up_stationery_le/'
def mean_img(d, n=16, cam='observation.images.top'):
    fs=sorted(glob.glob(R+d+'/videos/'+cam+'/chunk-000/*.mp4'))
    acc=None; c=0
    for f in fs[:n]:
        p=subprocess.run(['/usr/bin/ffmpeg','-loglevel','error','-i',f,'-vf','select=not(mod(n\\,300)),scale=320:240','-vsync','0','-f','image2pipe','-vcodec','png','-'],capture_output=True)
        buf=p.stdout
        # split pngs
        parts=buf.split(b'\x89PNG\r\n\x1a\n')
        for pt in parts[1:]:
            im=np.asarray(Image.open(io.BytesIO(b'\x89PNG\r\n\x1a\n'+pt)).convert('RGB'),dtype=np.float32)
            acc = im if acc is None else acc+im; c+=1
    return acc/c, c
a,ca=mean_img('batch_success_361')
b,cb=mean_img('batch_fail_72')
print('frames',ca,cb)
print('mean RGB succ', a.mean((0,1)), 'fail', b.mean((0,1)))
diff=np.abs(a-b).mean(2)
print('mean |diff| overall %.2f  p95 %.2f'%(diff.mean(),np.percentile(diff,95)))
# row profile of difference
rows=diff.mean(1)
print('top rows(0-40) diff %.2f, mid(80-160) %.2f, bottom(200-240) %.2f'%(rows[:40].mean(),rows[80:160].mean(),rows[200:].mean()))
Image.fromarray(np.uint8(np.clip(np.concatenate([a,b,np.stack([diff*3]*3,-1)],1),0,255))).save('/tmp/claude-1000/-home-kewei-YING-robot-data-platform/6bb03cdc-6944-4c6c-a95d-b8470ff79220/scratchpad/meanimg.png')
