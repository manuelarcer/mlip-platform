# UMA install recipe

_Last verified: pending first-run validation_

Package: [`fairchem-core`](https://github.com/FAIR-Chem/fairchem) — supplies the `FAIRChemCalculator` used for any `uma-*` tag.

> **Don't install `fairchem-core` into an env that already has `mace-torch`, `sevenn`, or `chgnet`.** The torch / torch_geometric pins collide. Create a fresh env per [ADR 0001](../adr/0001-per-mlip-envs.md).

## Conda recipe (preferred for HPC)

```bash
conda create -n mlip-uma python=3.11 -y
conda activate mlip-uma

# Pick ONE torch line depending on your hardware.
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1
# pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU

pip install fairchem-core ase
pip install -e /path/to/mlip_platform   # or: pip install mlip-platform
```

## Venv recipe (preferred for local dev)

```bash
python3.11 -m venv .venv-uma
source .venv-uma/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install fairchem-core ase
pip install -e /path/to/mlip_platform
```

## HuggingFace auth (required, one-time)

UMA weights are hosted on HuggingFace behind a gated license. **Skipping this step means the first run fails with a 401/403.**

1. Create a HuggingFace account if you don't have one.
2. Visit the [UMA model page](https://huggingface.co/facebook/UMA) and click **Agree and access repository**. Wait for the access grant (usually immediate).
3. Generate a read token at <https://huggingface.co/settings/tokens>.
4. Log in locally:

   ```bash
   pip install huggingface_hub
   huggingface-cli login   # paste the token when prompted
   ```

5. (HPC) If your compute nodes don't have outbound HTTPS, pre-fetch the weights on a login node:

   ```bash
   python -c "from fairchem.core import pretrained_mlip; pretrained_mlip.get_predict_unit('uma-s-1p2', device='cpu')"
   ```

   The download lands in `~/.cache/huggingface/`. Make sure this path is on a shared filesystem the compute nodes can read.

## Sanity check

```bash
python -c "from fairchem.core import pretrained_mlip, FAIRChemCalculator; p = pretrained_mlip.get_predict_unit('uma-s-1p2', device='cpu'); FAIRChemCalculator(p, task_name='omat'); print('OK')"
```

---

## <a id="uma-s-1p2"></a>Tag: `uma-s-1p2` (current default)

Base recipe above. The `--uma-task` CLI flag selects the head (`omat` / `oc20` / `omol` / `odac` — default `omat`). See `docs/UMA_USAGE_GUIDE.md` for task-head guidance.

## Other `uma-*` tags

Any `uma-*` tag is forwarded to `pretrained_mlip.get_predict_unit(tag)` unchanged. The same env recipe applies; only the weights differ (each tag triggers its own HF download on first use).
