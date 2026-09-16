# GDN-2 RMSNorm2 + W1/W2 + SwiGLU decode unit

This A5/CCE unit implements the fixed released-model `B=T=1`, `D=2304`,
`I=6208` boundary. Two vector subblocks compute RMSNorm and hand a BF16
normalized vector to cube; runtime keeps the paired W1/W2 stream and SwiGLU
epilogue in the same launch. The BF16 normalized-input, GEMV-output and
SiLU-output boundaries are explicit. W3 is intentionally separate.

Use the Ascriptor revision pinned by this repository, then run:

```sh
python run.py reference --case all --output tmp/reference
python run.py check --case random --launcher sim --output tmp/sim
python run.py check --case random --launcher pipesim --output tmp/pipesim
python run.py check --case random --launcher aclnn --backend cce --output tmp/board
```

Simulation establishes formula coverage and schedule legality only. Hardware
latency is the performance authority; in particular, an N=64 schedule looked
better in pipesim but lost on silicon, while compact FIX publication was the
hardware win that pipesim did not price.
