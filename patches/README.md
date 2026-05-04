# Patches

Local patches applied to the `fairseq_repo` submodule (pinned to facebookresearch/fairseq @ `5175fd5`) so it builds and runs under the modern stack used in this fork (Python 3.10, PyTorch 2.9.1+cu130, NumPy 1.26).

The submodule itself still points at the upstream commit; we cannot push these changes back to facebookresearch/fairseq, so they live here as patch files instead.

## How to apply

After `git submodule update --init --recursive`:

```bash
cd fairseq_repo
git apply ../patches/fairseq-5175fd-pt29-compat.patch
cd ..
```

If you ever need to refresh the patch:

```bash
cd fairseq_repo
git diff HEAD > ../patches/fairseq-5175fd-pt29-compat.patch
```

## What the patch fixes

- `fairseq/data/indexed_dataset.py`: `np.float` → `np.float32` (size 4 dtype, matching the `_element_sizes` declaration). Removed in NumPy 1.20+.
- `fairseq/data/data_utils.py`: `np.int` → `np.int64` for `np.fromiter` dtype.
- `fairseq/modules/dynamic_crf_layer.py`: `np.float("inf")` → `float("inf")`.
- `fairseq/checkpoint_utils.py`: `torch.load(..., weights_only=False)` for both `load_checkpoint_to_cpu` and `load_ema_from_checkpoint`. PyTorch 2.6+ flipped the default to `True`, which rejects the `argparse.Namespace` / OmegaConf objects that fairseq stores in its checkpoints.

Custom CUDA kernel (`NAST/models/torch_imputer/imputer.cu`) is patched directly in this repo (THCudaCheck → C10_CUDA_CHECK shim), so it does not need a separate patch file.
