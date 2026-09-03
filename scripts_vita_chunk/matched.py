import numpy as np
SP = "/tmp/claude-1000/-home-kewei-YING-lerobot-vlahost/45eaca5e-4060-47c9-ad96-fb29acfbaf35/scratchpad"
A = np.load(f"{SP}/A.npy")   # (93, 8, 16)

# path/net with the same "moving joints only" filter used on the training data
path = np.abs(np.diff(A, axis=1)).sum(axis=1)
net = np.abs(A[:, -1, :] - A[:, 0, :])
keep = net > 1e-4
r = path[keep] / net[keep]
print(f"VITA 8-step path/net (moving joints only): median={np.median(r):.2f}  p90={np.percentile(r,90):.2f}")

# cross-joint correlation of the de-trended residual
t = np.arange(A.shape[1])
R = np.empty_like(A)
for c in range(A.shape[0]):
    for j in range(A.shape[2]):
        k, b = np.polyfit(t, A[c, :, j], 1)
        R[c, :, j] = A[c, :, j] - (k * t + b)
flat = R.reshape(-1, 16)
C = np.corrcoef(flat.T)
iu = np.triu_indices(16, 1)
v = C[iu]
print(f"cross-joint residual corr: mean={v.mean():+.3f}  mean|r|={np.abs(v).mean():.3f}  frac|r|>0.5={np.mean(np.abs(v)>0.5)*100:.0f}%")
