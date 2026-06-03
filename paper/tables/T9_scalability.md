| Test samples (Khalifa) | Wall-clock (s) | ms / sample | Peak Python memory (MB) | Process RSS (MB) |
|---|---|---|---|---|
| 50 | 0.328 | 6.552 | 101.988 | 683.800 |
| 100 | 0.419 | 4.195 | 101.988 | 770.100 |
| 250 | 0.666 | 2.665 | 101.988 | 770.700 |
| 500 | 1.096 | 2.192 | 101.988 | 770.700 |
| 1000 | 1.912 | 1.912 | 101.988 | 770.700 |
| 2000 | 3.561 | 1.780 | 101.988 | 773.800 |

_Cost vs candidate-course vocabulary / graph size (N=300 samples): khalifa (vocab 59): 2.51 ms/sample; aus (vocab 61): 2.64 ms/sample; unc (vocab 74): 2.76 ms/sample. Model load 0.02s for 3 policies (RSS +4 MB); peak Python memory is flat in N. CPU-only; wall-clock is hardware-dependent. [src] scalability.json._
