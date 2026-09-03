d, ff, K = 512, 3200, 100
for name, hw, ncam in [("480x640", (15, 20), 3), ("240x320", (8, 10), 3), ("480x640", (15, 20), 1)]:
    h, w = hw
    N = 2 + ncam * h * w
    enc = 4 * N * d * d * 2 + 2 * N * N * d * 2 + 2 * N * d * ff * 2
    dec_self = 4 * K * d * d * 2 + 2 * K * K * d * 2
    dec_cross = (K * d * d * 2) * 2 + (2 * N * d * d * 2) + 2 * K * N * d * 2
    dec_ffn = 2 * K * d * ff * 2
    dec = dec_self + dec_cross + dec_ffn
    vae = 4 * (K + 2) * d * d * 2 + 2 * (K + 2) ** 2 * d * 2 + 2 * (K + 2) * d * ff * 2
    # resnet18 @224 = 1.82 GFLOP (2*MACs)
    backbone = 1.82e9 * (h * 32) * (w * 32) / (224 * 224) * ncam
    print(f"{name} ncam={ncam} N={N}")
    print(f"  backbone total     {backbone/1e9:8.2f} GFLOP")
    print(f"  obs-encoder /layer {enc/1e9:8.2f} GFLOP   x4 = {4*enc/1e9:7.2f}")
    print(f"  decoder     /layer {dec/1e9:8.2f} GFLOP   x7 = {7*dec/1e9:7.2f}")
    print(f"     of which cross-attn KV proj over memory: {(2*N*d*d*2)/1e9:.3f}")
    print(f"  vae-encoder /layer {vae/1e9:8.3f} GFLOP (train only)")
    print(f"  infer total (enc4+dec7+bb) = {(backbone+4*enc+7*dec)/1e9:.2f} GFLOP")
