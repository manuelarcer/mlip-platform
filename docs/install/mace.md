# MACE install recipe

_Last verified: pending first-run validation_

Package: [`mace-torch`](https://github.com/ACEsuit/mace) — supplies the calculators for `mace`, `mace-mh-0`, and `mace-mh-1`.

> **Don't install MACE into an env that already has `fairchem-core`, `sevenn`, or `chgnet`.** The torch / e3nn pins collide. Create a fresh env per [ADR 0001](../adr/0001-per-mlip-envs.md).

## Conda recipe (preferred for HPC)

```bash
conda create -n mlip-mace python=3.11 -y
conda activate mlip-mace

# Pick ONE torch line depending on your hardware.
# CUDA 12.1 (typical HPC GPU node):
pip install torch --index-url https://download.pytorch.org/whl/cu121
# CPU-only (laptops, login nodes without GPU):
# pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install mace-torch ase
pip install -e /path/to/mliprun   # or: pip install mliprun
```

## Venv recipe (preferred for local dev)

```bash
python3.11 -m venv .venv-mace
source .venv-mace/bin/activate

# Pick ONE torch line as above.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install mace-torch ase
pip install -e /path/to/mliprun
```

## Sanity check

```bash
python -c "from mace.calculators import mace_mp; c = mace_mp(model='medium', device='cpu'); print('OK')"
```

The first run downloads the MACE-MP-0 medium checkpoint into `~/.cache/mace/`.

---

## <a id="mace"></a>Tag: `mace` (MACE-MP-0 medium)

The base recipe above is everything you need. No extra steps.

Weights: auto-downloaded by `mace_mp()` on first calculator instantiation into `~/.cache/mace/`. Cache the file once on a node with internet if your compute nodes are air-gapped.

## <a id="mace-mh-0"></a>Tag: `mace-mh-0` (multi-head foundation, prerelease)

Base recipe above, plus the checkpoint download. The platform's `_ensure_mace_foundation_checkpoint` helper handles this lazily on first run; to pre-fetch:

```bash
mkdir -p ~/.cache/mace
curl -L -o ~/.cache/mace/mace-mh-0.model \
  https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-0.model
```

Use the `--mace-head` CLI flag to select a head (default `omat_pbe`). See `MACE_HEAD_HELP` in `src/mliprun/cli/utils.py` for the head menu.

## <a id="mace-mh-1"></a>Tag: `mace-mh-1` (multi-head foundation)

Same as `mace-mh-0`, different checkpoint:

```bash
mkdir -p ~/.cache/mace
curl -L -o ~/.cache/mace/mace-mh-1.model \
  https://github.com/ACEsuit/mace-foundations/releases/download/mace_mh_1/mace-mh-1.model
```
