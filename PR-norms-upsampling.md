# Add GroupNorm, InstanceNorm & Upsampling Layers (Nearest, Bilinear, PixelShuffle)

## Summary

- **GroupNorm**: Saves per-group `(mean, var)` during the `save=True` forward pass, expanded to per-channel shape for broadcasting. Main pass: `B * (weight / sqrt(var + eps))`. Bias pass: full affine with mean subtraction. Covers Stable Diffusion U-Net, YOLO, Mask R-CNN, FPN.
- **InstanceNorm**: Implemented as `nn.GroupNorm(num_groups=num_channels)` so the matrix computer handles it with zero extra code. Covers style transfer, CycleGAN, Pix2Pix.
- **Upsampling (Nearest, Bilinear)**: Uses `nn.Upsample` directly. Purely linear operations — applied directly in both main and bias passes with no activation ratios needed.
- **PixelShuffle**: Uses `nn.PixelShuffle` directly. Pure channel-to-spatial permutation — applied directly like AvgPool/Flatten.

## Files Changed

| File | Change |
|------|--------|
| `knowledgematrix/neural_net.py` | Added `groupnorm()`, `instancenorm()`, `upsample()`, `pixel_shuffle()` builder methods; added GroupNorm save block with per-group stats; added `nn.Upsample`/`nn.PixelShuffle` to pass-through; added `_has_groupnorm()` helper; updated `get_matrix_shape()` |
| `knowledgematrix/matrix_computer.py` | Added GroupNorm branches in main and bias passes; added `nn.Upsample`/`nn.PixelShuffle` to pass-through in both passes; added `_has_groupnorm()` to bias trigger condition |
| `extra/tests/groupnorm.py` | New: GroupNormCNN (8-group) and InstanceNormCNN tests |
| `extra/tests/upsample.py` | New: NearestUpsampleNet, BilinearUpsampleNet, PixelShuffleNet tests |
| `TO-ADD.md` | Marked items 5 and 6 as DONE |

## Architectures Unlocked

- **GroupNorm**: Stable Diffusion U-Net, YOLO v4-v8, Mask R-CNN, FPN
- **InstanceNorm**: Style transfer networks, CycleGAN, Pix2Pix
- **Upsampling**: U-Net (alternative to ConvTranspose2d), super-resolution (ESPCN), FPN

## Test Results

All tests pass. Knowledge matrix sum matches forward pass output at machine epsilon precision.

### GroupNorm (`extra/tests/groupnorm.py`) -- 5 tests

| Test | Input Shape | Classes | Batch | Diff | Time |
|------|------------|---------|-------|------|------|
| 1 | (1, 16, 16) | 7 | 2 | 3.58e-16 | 0.40s |
| 2 | (1, 18, 18) | 4 | 1 | 8.58e-16 | 0.37s |
| 3 | (3, 19, 19) | 20 | 6 | 1.21e-15 | 0.88s |
| 4 | (3, 23, 23) | 5 | 2 | 6.11e-16 | 2.61s |
| 5 | (1, 18, 18) | 9 | 2 | 8.51e-16 | 0.31s |

### InstanceNorm (`extra/tests/groupnorm.py`) -- 5 tests

| Test | Input Shape | Classes | Batch | Diff | Time |
|------|------------|---------|-------|------|------|
| 1 | (1, 27, 27) | 5 | 6 | 1.92e-16 | 0.39s |
| 2 | (3, 18, 18) | 8 | 1 | 2.28e-16 | 0.84s |
| 3 | (1, 29, 29) | 11 | 4 | 6.91e-16 | 0.66s |
| 4 | (3, 22, 22) | 2 | 5 | 1.39e-16 | 0.68s |
| 5 | (3, 23, 23) | 12 | 4 | 4.21e-16 | 0.87s |

### Nearest Upsample (`extra/tests/upsample.py`) -- 5 tests

| Test | Input Shape | Classes | Batch | Diff | Time |
|------|------------|---------|-------|------|------|
| 1 | (3, 18, 18) | 9 | 4 | 5.81e-17 | 0.37s |
| 2 | (3, 18, 18) | 5 | 6 | 2.40e-17 | 0.24s |
| 3 | (3, 28, 28) | 14 | 6 | 2.62e-17 | 1.10s |
| 4 | (3, 29, 29) | 11 | 2 | 5.19e-17 | 2.17s |
| 5 | (3, 16, 16) | 18 | 8 | 1.37e-16 | 0.16s |

### Bilinear Upsample (`extra/tests/upsample.py`) -- 5 tests

| Test | Input Shape | Classes | Batch | Diff | Time |
|------|------------|---------|-------|------|------|
| 1 | (1, 17, 17) | 4 | 6 | 3.10e-17 | 0.09s |
| 2 | (3, 32, 32) | 17 | 4 | 6.86e-17 | 2.26s |
| 3 | (3, 27, 27) | 14 | 8 | 9.94e-17 | 0.93s |
| 4 | (1, 27, 27) | 14 | 6 | 6.55e-17 | 0.35s |
| 5 | (1, 32, 32) | 2 | 6 | 3.47e-17 | 0.59s |

### PixelShuffle (`extra/tests/upsample.py`) -- 5 tests

| Test | Input Shape | Classes | Batch | Diff | Time |
|------|------------|---------|-------|------|------|
| 1 | (3, 16, 16) | 4 | 4 | 3.47e-17 | 0.51s |
| 2 | (1, 32, 32) | 18 | 6 | 8.30e-17 | 1.64s |
| 3 | (1, 29, 29) | 17 | 4 | 1.01e-16 | 2.32s |
| 4 | (1, 20, 20) | 12 | 1 | 8.44e-17 | 0.77s |
| 5 | (1, 20, 20) | 20 | 5 | 6.44e-17 | 0.32s |

### Regression Tests

**MLP (with residual connections):** 10/10 passed, max diff 3.37e-16
**SmallCNN (with residual connections):** 5/5 passed, max diff 1.92e-16
**RMSNorm:** 5/5 passed, max diff 1.21e-15
