# SevenNet install recipe

_Last verified: pending first-run validation_

Package: [`sevenn`](https://github.com/MDIL-SNU/SevenNet) — supplies the `SevenNetCalculator` used for the `7net-mf-ompa` tag.

> **Don't install `sevenn` into an env that already has `mace-torch`, `fairchem-core`, or `chgnet`.** The torch / `torch_geometric` pins collide. Create a fresh env per [ADR 0001](../adr/0001-per-mlip-envs.md).

## Conda recipe (preferred for HPC)

```bash
conda create -n mlip-sevenn python=3.11 -y
conda activate mlip-sevenn

# Pick ONE torch line depending on your hardware.
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU

# torch_geometric is required by sevenn; install it after torch.
pip install torch_geometric
pip install sevenn ase
pip install -e /path/to/mlip_platform   # or: pip install mlip-platform
```

## Venv recipe (preferred for local dev)

```bash
python3.11 -m venv .venv-sevenn
source .venv-sevenn/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch_geometric
pip install sevenn ase
pip install -e /path/to/mlip_platform
```

## Sanity check

```bash
python -c "from sevenn.calculator import SevenNetCalculator; SevenNetCalculator('7net-mf-ompa', modal='mpa', device='cpu'); print('OK')"
```

Weights ship inside the `sevenn` wheel — no separate download. No HF auth required.

---

## Tag: `7net-mf-ompa`

The only tag the platform currently exposes for SevenNet. The `modal='mpa'` argument selects the MPA (Materials Project + Alexandria) head.
