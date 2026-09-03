import numpy as np
SP="/tmp/claude-1000/-home-kewei-YING-robot-data-platform/3f79e71e-0bc0-4e0e-9ea6-e3c85a7c5058/scratchpad"
A=np.load(f"{SP}/A100k.npy"); S=np.load(f"{SP}/S100k.npy")
JOINTS=([f"left_arm_joint_{i}" for i in range(1,8)]+[f"right_arm_joint_{i}" for i in range(1,8)]+["left_gripper","right_gripper"])
C=A.shape[0]

# ---- Is the seam jump WITH the motion (lag/catch-up) or AGAINST it (oscillation)? ----
v_prev = A[:-1,-1,:]-A[:-1,-2,:]          # velocity at end of chunk k
seam   = A[1:,0,:]-A[:-1,-1,:]            # jump across the seam
v_next = A[1:,1,:]-A[1:,0,:]              # velocity at start of chunk k+1
m = np.abs(v_prev)>1e-4
print("=== seam direction vs motion ===")
print(f"seam agrees with prev-chunk velocity: {np.mean(np.sign(seam[m])==np.sign(v_prev[m]))*100:.1f}% of (chunk,joint)")
print(f"seam agrees with next-chunk velocity: {np.mean(np.sign(seam[m])==np.sign(v_next[m]))*100:.1f}%")
print(f"prev-vel vs next-vel same sign:       {np.mean(np.sign(v_prev[m])==np.sign(v_next[m]))*100:.1f}%")
print(f"|seam| / |v_prev| ratio  median={np.median(np.abs(seam[m])/np.abs(v_prev[m])):.2f}")

# ---- Where the arm actually was vs where the previous chunk said it'd be ----
pred_at_seam = A[:-1,-1,:]     # chunk k's last commanded pos
actual_next  = S[1:,:]         # measured state when chunk k+1 was requested
print(f"\n|state(k+1) - last_action(k)| mean={np.abs(actual_next-pred_at_seam).mean():.5f} max={np.abs(actual_next-pred_at_seam).max():.5f}")
print(f"|action0(k+1) - state(k+1)|   mean={np.abs(A[1:,0,:]-S[1:,:]).mean():.5f} max={np.abs(A[1:,0,:]-S[1:,:]).max():.5f}")

# ---- Velocity profile inside chunk, normalized per chunk (shape, not magnitude) ----
d=np.diff(A,axis=1)                       # (C,7,16)
sp=np.abs(d).sum(axis=2)                  # (C,7) L1 speed per step
sp_n=sp/np.maximum(sp.mean(axis=1,keepdims=True),1e-9)
print("\n=== normalized speed profile inside chunk (1.0 = chunk mean) ===")
for i in range(7): print(f"  step {i}->{i+1}: {sp_n[:,i].mean():.3f}")
print(f"  peak/edge ratio = {sp_n[:,3].mean()/((sp_n[:,0].mean()+sp_n[:,6].mean())/2):.2f}")

# ---- Executed stream: speed modulation at chunk rate ----
stream=A.reshape(-1,16); sd=np.abs(np.diff(stream,axis=0)).sum(axis=1)
ph=np.array([sd[i::8].mean() for i in range(8)])
print("\n=== executed-stream speed by phase within the 8-step cycle (incl. seam at phase 7) ===")
for i,v in enumerate(ph): print(f"  phase {i}: {v:.5f}" + ("   <- seam" if i==7 else ""))
print(f"  modulation (max/min) = {ph.max()/ph.min():.2f}x")

# ---- gripper detail ----
print("\n=== grippers ===")
for i,name in [(14,"left_gripper"),(15,"right_gripper")]:
    a=A[:,:,i]
    print(f"{name}: range [{a.min():+.4f},{a.max():+.4f}] span={a.max()-a.min():.4f}  intra|d|mean={np.abs(np.diff(a,axis=1)).mean():.5f}")
    print(f"    state range [{S[:,i].min():+.4f},{S[:,i].max():+.4f}]")
