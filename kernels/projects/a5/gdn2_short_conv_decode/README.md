# `a5.gdn2_short_conv_decode`

Fixed-shape BF16 packed q/k/v depthwise width-four short convolution for one
GDN-2 decode token. The unit also applies SiLU and writes the next cache in the
same launch. `contract.json` is the authoritative ABI and validation record.

With the pinned Ascriptor 0.1.0 environment:

```bash
python run.py reference --output tmp/reference
python run.py check --launcher sim --output tmp/sim
python run.py check --launcher pipesim --output tmp/pipesim
python run.py emit --backend cce --output tmp/source
```
