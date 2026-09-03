# ACT-DiT under the deployment-faithful harness

Data and runners for
[`../../act_dit/act_dit-flowmatching-deployed-eval-2026-08.md`](../../act_dit/act_dit-flowmatching-deployed-eval-2026-08.md).

The harness itself is `../scripts_act_eval_test_fix/offline_chunk_eval.py` — unmodified, nothing
is forked here. Method, metric definitions and the deploy rewrite it reproduces are documented in
[`../scripts_act_eval_test_fix/README.md`](../scripts_act_eval_test_fix/README.md).

```bash
./run_eval.sh                     # act_dit 08-27, 4 conditions, ~6.5 min GPU
CKPT=<path> TAG=<name> ./run_eval.sh   # any other checkpoint
./run_compare.sh                  # act_dit 08-20 replicate + ACT baseline on eval53, ~4 min
python3 summarize.py *.json       # headline table
```

Both runners `export PYTHONPATH=/home/kewei/YING/lerobot_vlahost/src`: the `lerobot` installed in
`/opt/robot-platform/train-venv` on mgmt01 predates `ACTDiTConfig.use_ema`, and loading an EMA
checkpoint without it dies with `DecodingError: The fields use_ema, ema_decay are not valid`.
