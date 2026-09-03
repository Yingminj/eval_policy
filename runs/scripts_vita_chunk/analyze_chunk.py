"""Parse record_chunk.txt into (n_chunks, 8, 16) actions + (n_chunks, 16) states."""
import ast
import re
import numpy as np

TXT = "/home/kewei/YING/lerobot_vlahost/record_chunk.txt"
JOINTS = (
    [f"left_arm_joint_{i}" for i in range(1, 8)]
    + [f"right_arm_joint_{i}" for i in range(1, 8)]
    + ["left_gripper", "right_gripper"]
)
KEYS = [f"{j}.pos" for j in JOINTS]

states, chunks = [], []
cur = None
for line in open(TXT):
    line = line.strip()
    if line.startswith("=== Chunk"):
        if cur:
            chunks.append(cur)
        cur = []
    elif line.startswith("Robot State"):
        states.append(ast.literal_eval(line.split(":", 1)[1].strip()))
    elif line.startswith("Action "):
        d = ast.literal_eval(line.split(":", 1)[1].strip())
        cur.append([d[k] for k in KEYS])
if cur:
    chunks.append(cur)

A = np.array(chunks)          # (C, 8, 16)
S = np.array(states)          # (C, 16)
np.save("/tmp/claude-1000/-home-kewei-YING-lerobot-vlahost/45eaca5e-4060-47c9-ad96-fb29acfbaf35/scratchpad/A.npy", A)
np.save("/tmp/claude-1000/-home-kewei-YING-lerobot-vlahost/45eaca5e-4060-47c9-ad96-fb29acfbaf35/scratchpad/S.npy", S)
print("actions:", A.shape, "states:", S.shape)

np.set_printoptions(precision=4, suppress=True, linewidth=200)

# --- 1. Intra-chunk step-to-step motion vs inter-chunk seam jump ---
intra = np.abs(np.diff(A, axis=1))                 # (C, 7, 16) step->step inside a chunk
seam = np.abs(A[1:, 0, :] - A[:-1, -1, :])         # (C-1, 16) last action -> next chunk's first
print("\n--- magnitude of motion (rad) ---")
print(f"intra-chunk |step delta|   mean={intra.mean():.5f}  p95={np.percentile(intra,95):.5f}  max={intra.max():.5f}")
print(f"chunk-seam  |jump|         mean={seam.mean():.5f}  p95={np.percentile(seam,95):.5f}  max={seam.max():.5f}")
print(f"seam / intra ratio (mean): {seam.mean()/intra.mean():.2f}x")

# --- 2. Is the first action of each chunk just the current state? ---
d0 = np.abs(A[:, 0, :] - S)
print(f"\n|action0 - robot_state| mean={d0.mean():.6f} max={d0.max():.6f}")

# --- 3. Direction reversals within a chunk (the signature of vibration) ---
d = np.diff(A, axis=1)                              # (C, 7, 16)
sign_flips = (np.sign(d[:, 1:, :]) != np.sign(d[:, :-1, :])) & (np.abs(d[:, 1:, :]) > 1e-4)
print(f"\nintra-chunk direction reversals: {sign_flips.mean()*100:.1f}% of consecutive step pairs")

# --- 4. Per-joint breakdown ---
print("\n--- per joint (rad) ---")
print(f"{'joint':<22}{'intra_mean':>12}{'intra_max':>12}{'seam_mean':>12}{'seam_max':>12}{'reversal%':>11}")
for i, j in enumerate(JOINTS):
    print(f"{j:<22}{intra[:,:,i].mean():>12.5f}{intra[:,:,i].max():>12.5f}"
          f"{seam[:,i].mean():>12.5f}{seam[:,i].max():>12.5f}{sign_flips[:,:,i].mean()*100:>10.1f}%")

# --- 5. Overall drift: how far does the arm actually travel? ---
span = S.max(axis=0) - S.min(axis=0)
print("\nstate range over the whole recording (rad):")
for i, j in enumerate(JOINTS):
    print(f"  {j:<22}{span[i]:.5f}")
