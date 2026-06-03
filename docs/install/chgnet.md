# CHGNet install recipe

_Last verified: pending first-run validation_

Package: [`chgnet`](https://github.com/CederGroupHub/chgnet) — supplies the `CHGNetCalculator` used for the `chgnet` tag.

> **Don't install `chgnet` into an env that already has `mace-torch`, `fairchem-core`, or `sevenn`.** The torch pins are looser than the others but still collide in practice. Create a fresh env per [ADR 0001](../adr/0001-per-mlip-envs.md).

## Conda recipe (preferred for HPC)

```bash
conda create -n mlip-chgnet python=3.11 -y
conda activate mlip-chgnet

# Pick ONE torch line depending on your hardware.
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU

pip install chgnet ase
pip install -e /path/to/mlip_platform   # or: pip install mlip-platform
```

## Venv recipe (preferred for local dev)

```bash
python3.11 -m venv .venv-chgnet
source .venv-chgnet/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install chgnet ase
pip install -e /path/to/mlip_platform
```

## Sanity check

```bash
python -c "from chgnet.model.dynamics import CHGNetCalculator; CHGNetCalculator(use_device='cpu'); print('OK')"
```

Weights ship inside the `chgnet` wheel — no separate download. No HF auth required.

---

## Tag: `chgnet`

The only tag the platform exposes. Trained on the Materials Project relaxation trajectories; fastest cold start of the four MLIPs.
