import numpy as np

SP = "/tmp/claude-1000/-home-kewei-YING-lerobot-vlahost/45eaca5e-4060-47c9-ad96-fb29acfbaf35/scratchpad"
A = np.load(f"{SP}/A.npy")   # (93, 8, 16)
S = np.load(f"{SP}/S.npy")
np.set_printoptions(precision=4, suppress=True, linewidth=200)

# --- smoothness: path length vs net displacement, per chunk per joint ---
path = np.abs(np.diff(A, axis=1)).sum(axis=1)        # (C,16) total travel inside the chunk
net = np.abs(A[:, -1, :] - A[:, 0, :])               # (C,16) net displacement
ratio = path / np.maximum(net, 1e-9)
print(f"path/net ratio: median={np.median(ratio):.2f}  mean={ratio.mean():.2f}")
print("  (1.0 = perfectly monotonic; >2 means it doubles back more than it advances)")

# --- lag-1 autocorrelation of the step deltas: negative == zigzag ---
d = np.diff(A, axis=1)                               # (C,7,16)
ac = []
for c in range(d.shape[0]):
    for j in range(d.shape[2]):
        x = d[c, :, j]
        x = x - x.mean()
        if np.std(x) < 1e-9:
            continue
        ac.append(np.corrcoef(x[:-1], x[1:])[0, 1])
print(f"\nlag-1 autocorr of step deltas: mean={np.nanmean(ac):.3f}")
print("  (smooth trajectory -> near +1; white noise -> 0; alternating zigzag -> -0.5)")

# --- split each chunk into a linear trend + residual ---
t = np.arange(A.shape[1])
resid_amp, trend_amp = [], []
for c in range(A.shape[0]):
    for j in range(A.shape[2]):
        y = A[c, :, j]
        k, b = np.polyfit(t, y, 1)
        fit = k * t + b
        resid_amp.append(np.abs(y - fit).mean())
        trend_amp.append(np.abs(fit[-1] - fit[0]))
resid_amp = np.array(resid_amp); trend_amp = np.array(trend_amp)
print(f"\nper-chunk linear trend span: mean={trend_amp.mean():.5f} rad")
print(f"residual around that trend:  mean={resid_amp.mean():.5f} rad")
print(f"noise/signal = {resid_amp.mean()/trend_amp.mean():.2f}")

# --- what the arm actually experiences: concatenated executed stream ---
stream = A.reshape(-1, 16)
sd = np.diff(stream, axis=0)
print(f"\nexecuted stream |step delta|: mean={np.abs(sd).mean():.5f} p99={np.percentile(np.abs(sd),99):.5f} max={np.abs(sd).max():.5f}")
print(f"at 30 fps that is {np.abs(sd).mean()*30:.3f} rad/s mean, {np.abs(sd).max()*30:.3f} rad/s peak")

# --- per-step position within the chunk: is the noise uniform or edge-heavy? ---
print("\nmean |delta| by position inside the chunk (step i -> i+1):")
for i in range(d.shape[1]):
    print(f"  {i}->{i+1}: {np.abs(d[:, i, :]).mean():.5f}")
