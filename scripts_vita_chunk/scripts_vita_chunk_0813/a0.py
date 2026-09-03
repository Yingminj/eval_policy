import numpy as np
SP="/tmp/claude-1000/-home-kewei-YING-robot-data-platform/3f79e71e-0bc0-4e0e-9ea6-e3c85a7c5058/scratchpad"
A=np.load(f"{SP}/A100k.npy"); S=np.load(f"{SP}/S100k.npy")
JOINTS=([f"left_arm_joint_{i}" for i in range(1,8)]+[f"right_arm_joint_{i}" for i in range(1,8)]+["left_gripper","right_gripper"])
d=np.abs(A[:,0,:]-S)
exact = d < 1e-6
print(f"action0 == robot_state exactly (<1e-6): {exact.mean()*100:.1f}% of {d.size} (chunk,joint) entries")
print(f"among non-exact: mean={d[~exact].mean():.5f} max={d[~exact].max():.5f}  count={(~exact).sum()}")
print("\nper joint: frac exact / mean dev when not exact")
for i,j in enumerate(JOINTS):
    ne=~exact[:,i]
    md=d[ne,i].mean() if ne.any() else 0.0
    print(f"  {j:<22} exact={exact[:,i].mean()*100:>5.1f}%  dev={md:.5f}")
print("\nper-chunk count of non-exact joints (first 30 chunks):")
print((~exact).sum(axis=1)[:30])
print("total chunks with 0 non-exact:", (( ~exact).sum(axis=1)==0).sum(), "of", A.shape[0])
