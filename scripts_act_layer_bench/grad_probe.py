"""Show that indexing the decoder stack with [0] (upstream ACT) leaves layers 1..N-1 untrained."""
import argparse
import sys

import torch

sys.path.insert(0, "/home/kewei/YING/ACT_rosbag2")
sys.path.insert(0, "/home/kewei/YING/ACT_rosbag2/detr")

from detr.main import get_args_parser
from detr.models import build_ACT_model

parser = argparse.ArgumentParser(parents=[get_args_parser()])
a = parser.parse_args([])
a.backbone, a.camera_names = "resnet18", ["top", "wrist_L", "wrist_R"]
a.hidden_dim, a.dim_feedforward, a.nheads = 512, 3200, 8
a.enc_layers, a.dec_layers, a.num_queries, a.action_dim = 4, 7, 100, 16
m = build_ACT_model(a).cuda().train()

img = torch.randn(2, 3, 3, 240, 320, device="cuda")
qpos = torch.randn(2, 16, device="cuda")
act = torch.randn(2, 100, 16, device="cuda")
is_pad = torch.zeros(2, 100, dtype=torch.bool, device="cuda")

for idx in (0, -1):
    m.zero_grad(set_to_none=True)
    # replicate DETRVAE.forward but with a configurable decoder-stack index
    import detr.models.detr_vae as dv

    orig = dv.DETRVAE.forward

    def patched(self, qpos, image, env_state, actions=None, is_pad=None, _i=idx):
        bs, _ = qpos.shape
        ae = self.encoder_action_proj(actions)
        qe = self.encoder_joint_proj(qpos).unsqueeze(1)
        ce = self.cls_embed.weight.unsqueeze(0).repeat(bs, 1, 1)
        ei = torch.cat([ce, qe, ae], axis=1).permute(1, 0, 2)
        pad = torch.cat([torch.full((bs, 2), False).cuda(), is_pad], axis=1)
        pe = self.pos_table.clone().detach().permute(1, 0, 2)
        eo = self.encoder(ei, pos=pe, src_key_padding_mask=pad)[0]
        li = self.latent_proj(eo)
        mu, logvar = li[:, : self.latent_dim], li[:, self.latent_dim :]
        z = dv.reparametrize(mu, logvar)
        lat = self.latent_out_proj(z)
        feats, poss = [], []
        for cam_id in range(len(self.camera_names)):
            f, p = self.backbones[0](image[:, cam_id])
            feats.append(self.input_proj(f[0]))
            poss.append(p[0])
        prop = self.input_proj_robot_state(qpos)
        hs = self.transformer(
            torch.cat(feats, axis=3), None, self.query_embed.weight,
            torch.cat(poss, axis=3), lat, prop, self.additional_pos_embed.weight,
        )[_i]
        return self.action_head(hs), self.is_pad_head(hs), [mu, logvar]

    dv.DETRVAE.forward = patched
    a_hat, _, _ = m(qpos, img, None, act, is_pad)
    torch.nn.functional.l1_loss(a_hat, act).backward()
    dv.DETRVAE.forward = orig

    norms = []
    for i, layer in enumerate(m.transformer.decoder.layers):
        g = layer.linear1.weight.grad
        norms.append(0.0 if g is None else g.norm().item())
    print(f"index [{_i if False else idx}] -> per-decoder-layer grad norm (linear1.weight):")
    print("   " + "  ".join(f"L{i}:{n:.3e}" for i, n in enumerate(norms)))
