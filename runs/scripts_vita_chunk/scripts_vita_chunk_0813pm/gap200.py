import numpy as np
SP="/home/kewei/.claude/jobs/b715062d/tmp"
A=np.load(f"{SP}/A200k.npy"); S=np.load(f"{SP}/S200k.npy")
g=np.abs(A[:,0,:]-S)
print("=== model |action0 - state| vs training-set |action[t]-state[t]| ===")
print(f"           model(200k raw)   training(batch_3)   ratio")
print(f"arms 0-13    {g[:,:14].mean():.5f}          0.02028          {g[:,:14].mean()/0.02028:.2f}x")
print(f"grippers     {g[:,14:].mean():.5f}          0.00012        {g[:,14:].mean()/0.00012:>6.0f}x")
print(f"all 16       {g.mean():.5f}          0.01776          {g.mean()/0.01776:.2f}x")

# tracking lag in time units
intra=np.abs(np.diff(A,axis=1)).mean()
lag=np.abs(S[1:]-A[:-1,-1,:]).mean()
print(f"\n=== tracking lag ===")
print(f"|state(k+1) - last_action(k)| = {lag:.5f} rad")
print(f"mean intra-chunk step        = {intra:.5f} rad/frame")
print(f"=> lag equals {lag/intra:.1f} frames = {lag/intra/30:.3f} s at 30 fps")

# seam decomposed: how much is 'catch up to where the arm is' vs excess
seam=A[1:,0,:]-A[:-1,-1,:]
catchup=S[1:]-A[:-1,-1,:]          # the part explained by the arm lagging
excess=seam-catchup                 # what the model adds on top
print(f"\n=== seam decomposition (mean |.|, rad) ===")
print(f"total seam jump          {np.abs(seam).mean():.5f}")
print(f"  explained by arm lag   {np.abs(catchup).mean():.5f}   (corr with seam: {np.corrcoef(seam.ravel(),catchup.ravel())[0,1]:+.3f})")
print(f"  residual (model offset) {np.abs(excess).mean():.5f}")

# gripper training target is an identity copy of the observation -> flag it
print(f"\n=== gripper channel caveat ===")
print(f"training |action-state| for grippers = 0.00012 rad  -> action column is an identity copy of the observation")
print(f"training lag-1 autocorr: arms +0.973 vs grippers +0.731 -> gripper target is intrinsically noisier")
