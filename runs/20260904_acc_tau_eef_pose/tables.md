### reproduction check (policy_raw mae vs the 09-03 report)

| run | horizon | this run | report | |
|---|---:|---:|---:|---|
| `pp_eef_state` | 50 | 0.03721 | 0.03721 | MATCH |
| `pp_eef_nostate` | 50 | 0.03916 | 0.03916 | MATCH |
| `pp_eef_act5` | 50 | 0.03796 | 0.03796 | MATCH |
| `acteef_533 (h60)` | 60 | 0.03689 | 0.03689 | MATCH |
| `acteef_533 (h50)` | 50 | 0.03443 | 0.03443 | MATCH |

### new: acc@tau, joint/EEF-vector space (fraction of steps inside tau*sigma)

| run | @0.1s | @0.25s | @0.5s | @1s | acc@0.25s decay 1 -> 50 |
|---|---:|---:|---:|---:|---|
| `pp_eef_state` | 0.483 | 0.757 | 0.914 | 0.983 | 0.880 -> 0.757 |
| `pp_eef_nostate` | 0.454 | 0.728 | 0.902 | 0.982 | 0.843 -> 0.728 |
| `pp_eef_act5` | 0.467 | 0.741 | 0.912 | 0.985 | 0.849 -> 0.741 |
| `acteef_533 (h60)` | 0.514 | 0.764 | 0.911 | 0.982 | 0.937 -> 0.764 |
| `acteef_533 (h50)` | 0.530 | 0.783 | 0.923 | 0.985 | 0.937 -> 0.783 |

### new: end-effector pose error (Euclidean / geodesic, both arms averaged)

| run | pos (mm) | rot (deg) | report per-axis (mm) | report per-axis (deg) | ratio |
|---|---:|---:|---:|---:|---:|
| `pp_eef_state` | 22.67 | 6.92 | 11.28 | 3.86 | 2.01x |
| `pp_eef_nostate` | 24.66 | 7.01 | 12.27 | 4.05 | 2.01x |
| `pp_eef_act5` | 22.91 | 6.76 | 11.38 | 3.93 | 2.01x |
| `acteef_533 (h60)` | 21.27 | 6.73 | 10.51 | 3.87 | 2.02x |
| `acteef_533 (h50)` | 19.81 | 6.24 | 9.78 | 3.61 | 2.03x |

### new: EEF acc@tau -- fraction of executed steps inside a real tolerance (left arm)

| run | 5mm+0.01rad | 10mm+0.025rad | 25mm+0.05rad | 50mm+0.1rad |
|---|---|---|---|---|
| `pp_eef_state` | 0.042 | 0.214 | 0.711 | 0.934 |
| `pp_eef_nostate` | 0.035 | 0.184 | 0.676 | 0.927 |
| `pp_eef_act5` | 0.049 | 0.185 | 0.701 | 0.949 |
| `acteef_533 (h60)` | 0.093 | 0.292 | 0.741 | 0.942 |
| `acteef_533 (h50)` | 0.101 | 0.316 | 0.772 | 0.956 |

nulls, for the same horizon as the run above them:
  pp_eef_state       hold_state  5mm+0.01rad=0.063  10mm+0.025rad=0.153  25mm+0.05rad=0.447  50mm+0.1rad=0.709
  pp_eef_nostate     hold_state  5mm+0.01rad=0.063  10mm+0.025rad=0.153  25mm+0.05rad=0.447  50mm+0.1rad=0.709
  pp_eef_act5        hold_state  5mm+0.01rad=0.063  10mm+0.025rad=0.153  25mm+0.05rad=0.447  50mm+0.1rad=0.709
  acteef_533 (h60)   hold_state  5mm+0.01rad=0.055  10mm+0.025rad=0.138  25mm+0.05rad=0.413  50mm+0.1rad=0.668
  acteef_533 (h50)   hold_state  5mm+0.01rad=0.063  10mm+0.025rad=0.153  25mm+0.05rad=0.447  50mm+0.1rad=0.709
