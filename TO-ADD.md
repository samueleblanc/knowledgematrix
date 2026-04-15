# Knowledge Matrix Library -- Features To Add

## Top 10 by Simplicity (Coding & Testing)

### 1. ~~Grouped & Dilated Convolutions~~ (DONE)
**Difficulty: Trivial (2-line change)** -- **Added in branch `grouped-dilated-transposed-conv`**

The current `F.conv2d` call in `matrix_computer.py` does not pass `groups` or `dilation`. Adding these two parameters unlocks depthwise separable convolutions, ResNeXt, MobileNet, EfficientNet, DeepLab, and WaveNet.

**What to change:**
```python
# matrix_computer.py line ~107, change:
B = F.conv2d(B, layer.weight, None, stride=layer.stride, padding=layer.padding)
# to:
B = F.conv2d(B, layer.weight, None, stride=layer.stride, padding=layer.padding,
             dilation=layer.dilation, groups=layer.groups)
```
Also expose `groups` and `dilation` params in the `conv()` / `conv1d()` builder methods in `neural_net.py`.

**Architectures unlocked:** MobileNet, EfficientNet, ResNeXt, ShuffleNet, DeepLab, WaveNet

---

### 2. ~~Transposed Convolutions (ConvTranspose2d / ConvTranspose1d)~~ (DONE)
**Difficulty: Easy (~25-30 lines)** -- **Added in branch `grouped-dilated-transposed-conv`**

ConvTranspose2d is a linear operation (the adjoint of Conv2d). It is the standard decoder layer for convolutional autoencoders, U-Nets, GANs, and super-resolution networks.

**What to change:**
- Add `conv_transpose()` builder method in `neural_net.py`
- Add `isinstance(layer, nn.ConvTranspose2d)` branch in `matrix_computer.py` using `F.conv_transpose2d(B, layer.weight, None, stride=..., padding=..., output_padding=...)`
- Same pattern as Conv2d -- no activation ratios needed

**Architectures unlocked:** Convolutional autoencoders, VAEs, U-Net, GANs (DCGAN, StyleGAN), super-resolution

---

### 3. RMSNorm
**Difficulty: Easy (~20 lines)**

RMSNorm is `x * weight / sqrt(mean(x^2) + eps)` -- a simplified LayerNorm without mean subtraction. Used by virtually all modern LLMs (LLaMA, Mistral, Gemma, Qwen).

**What to change:**
- Add `RMSNorm` module (or use `torch.nn.RMSNorm` from PyTorch 2.4+)
- Save `rms = sqrt(mean(x^2) + eps)` during the `save=True` forward pass (like `self.layernorms`)
- In matrix computer: `B = B * (weight / rms)` -- purely multiplicative, simpler than LayerNorm
- Add `_has_rmsnorm()` helper for bias column detection

**Architectures unlocked:** LLaMA 1/2/3, Mistral, Gemma, Qwen, T5, most post-2023 LLMs

---

### 4. Additional Activation Functions
**Difficulty: Easy (~15 lines total)**

Several common activations are missing. All are pointwise nonlinearities that work with the existing `post_act / pre_act` ratio trick.

| Activation | Formula | Used by |
|---|---|---|
| `ReLU6` | `min(max(x, 0), 6)` | MobileNet v1/v2 |
| `Softplus` | `log(1 + exp(x))` | Mamba, some normalizing flows |
| `Hardswish` | `x * relu6(x + 3) / 6` | MobileNet v3 |
| `Hardsigmoid` | `relu6(x + 3) / 6` | MobileNet v3, EfficientNet |
| `PReLU` | `max(0, x) + a * min(0, x)` | Some ResNets, face recognition |
| `CELU` | `max(0,x) + min(0, alpha*(exp(x/alpha)-1))` | Some research nets |

**What to change:** Add `isinstance` checks in the activation handling section of `neural_net.py` and `matrix_computer.py`.

---

### 5. GroupNorm & InstanceNorm
**Difficulty: Easy-Medium (~40 lines)**

GroupNorm divides channels into groups and normalizes per group. InstanceNorm is GroupNorm with groups=C. Both follow the same pattern as existing BatchNorm2d / LayerNorm support.

| Norm | Groups | Used by |
|---|---|---|
| GroupNorm | User-defined (e.g. 32) | Stable Diffusion, YOLO v4-v8, Mask R-CNN, FPN |
| InstanceNorm | C (one per channel) | Style transfer, CycleGAN, Pix2Pix |

**What to change:**
- Save `(mean, var)` per group during `save=True` forward pass
- Matrix computer: `B = B * (weight / sqrt(var_g + eps))` where `var_g` is broadcast per channel group
- Bias pass applies full affine transformation

**Architectures unlocked:** Stable Diffusion U-Net, YOLO, Mask R-CNN, style transfer networks

---

### 6. Upsampling Layers (Nearest, Bilinear, PixelShuffle)
**Difficulty: Easy (~30 lines)**

All three are linear operations:
- **Nearest:** repeats pixels -- `F.interpolate(B, scale_factor=r, mode='nearest')`
- **Bilinear:** weighted average of neighbors -- `F.interpolate(B, scale_factor=r, mode='bilinear')`
- **PixelShuffle:** reshuffles channels into spatial dims -- `F.pixel_shuffle(B, r)` (pure permutation)

**What to change:**
- Add `upsample()` and `pixel_shuffle()` builder methods in `neural_net.py`
- Apply corresponding `F.*` functions in the matrix computer (no activation ratios needed)

**Architectures unlocked:** U-Net (alternative to ConvTranspose2d), super-resolution (ESPCN), FPN

---

### 7. Feature Attribution Maps & Visualization
**Difficulty: Easy (~80 lines, no core changes)**

The knowledge matrix's unique selling point: since `output_j = sum_i A[j, i]`, column `A[j, :]` gives the *exact* contribution of each input element to output `j`. This is more principled than Grad-CAM (which is approximate).

**What to add (new file `knowledgematrix/visualization.py`):**
- `attribution_map(A, output_class, input_shape)` -- reshape column to `(C, H, W)`, sum channels for 2D heatmap
- `overlay_attribution(image, attribution)` -- overlay heatmap on input image
- `top_k_contributors(A, output_class, k)` -- identify most important input positions
- `per_class_comparison(A, classes)` -- side-by-side attribution for different classes
- `layer_contribution(A_layers)` -- bar chart of per-layer contribution magnitude

**Impact:** Turns the library from a computation tool into an interpretability toolkit

---

### 8. GQA / MQA (Grouped-Query & Multi-Query Attention)
**Difficulty: Easy-Medium (~40 lines)**

GQA uses fewer K/V heads than Q heads, with K/V shared across groups. MQA is the extreme case (1 K/V head). The existing `MultiHeadAttention` just needs a `num_kv_heads` parameter.

**What to change:**
- Add `num_kv_heads` parameter to `MultiHeadAttention.__init__()`
- K and V projections output `num_kv_heads * head_dim` instead of `num_heads * head_dim`
- Repeat K/V heads to match Q heads before attention computation
- Activation ratio approach is unchanged -- MHA is still treated as one nonlinear block

**Architectures unlocked:** LLaMA 2/3, Mistral, Gemma 2, Falcon

---

### 9. RoPE (Rotary Position Embeddings)
**Difficulty: Medium (~40-50 lines)**

RoPE applies position-dependent rotation matrices to pairs of dimensions in Q and K. At a fixed position, it is a linear transformation. Used by virtually all modern LLMs instead of additive positional embeddings.

**What to change:**
- Add `rope=True` option to `MultiHeadAttention`
- Precompute rotation matrices (cos/sin tables) based on `max_seq_len` and `head_dim`
- Apply rotation to Q and K inside `forward()` before computing attention scores
- No change needed in matrix computer -- the whole MHA block is still handled via activation ratios

**Architectures unlocked:** LLaMA, Mistral, Falcon, Gemma, CodeLlama, Qwen

---

### 10. Auto-Conversion from PyTorch nn.Module (`from_torch()`)
**Difficulty: Medium (~150 lines)**

The biggest adoption barrier: every architecture must be manually reimplemented as an `NN` subclass. A `from_torch()` utility would walk a standard `nn.Module` and automatically build the equivalent `NN`.

**What to add:**
- `NN.from_torch(module)` classmethod that recursively walks `module.named_modules()`
- Maps each leaf module to the corresponding `NN` builder method (Conv2d -> `self.conv()`, ReLU -> `self.relu()`, etc.)
- Start with sequential models (no residuals) covering ~80% of use cases
- Add `torch.fx` tracing for automatic residual connection detection in a second pass
- Load weights from the original module into the new NN

**Impact:** Would allow instant knowledge matrix computation for any compatible PyTorch model, including pretrained models from torchvision, timm, and potentially HuggingFace

---

## Honorable Mentions (Medium-Hard, High Impact)

| Feature | Difficulty | Why it matters |
|---|---|---|
| **SwiGLU activation** | Hard (~50 lines) | Used by LLaMA -- gated activation `SiLU(xW1) * xW2`, requires handling element-wise product of two input-dependent branches |
| **Concatenation-based skip connections** | Hard (~70 lines) | Needed for U-Net, DenseNet -- requires new residual mechanism for channel-concat vs addition |
| **Cross-attention** | Hard (~60 lines) | Needed for encoder-decoder models, Stable Diffusion -- Q from one stream, K/V from another |
| **Squeeze-and-Excitation blocks** | Hard (~80 lines) | EfficientNet -- input-dependent channel gating |
| **Mixed precision (float16/bfloat16)** | Easy (~20 lines) | Halves memory for large knowledge matrices |
| **Sparse matrix storage** | Medium (~50 lines) | ReLU networks produce many zeros; COO/CSR can save 5-50x memory |
| **LSTM / GRU** | Hard (~150+ lines) | Recurrent layers require unrolling + handling elementwise products of two input-dependent paths |
| **Mamba / SSM** | Very Hard (~200+ lines) | Input-dependent state transitions require new theoretical framework |
| **Mixture of Experts** | Hard (~100+ lines) | Branching + gate*expert product requires multi-path computation support |

## Architecture Unlock Roadmap

After implementing the top 10, here is what becomes possible:

| Architecture | Items Needed | Status After Top 10 |
|---|---|---|
| **GPT-2** | Causal mask (already works) | **Fully supported** |
| **BERT** | Learned PE + token select | **Nearly supported** (minor additions) |
| **MobileNet** | Grouped conv (#1) + ReLU6 (#4) | **Fully supported** |
| **ResNeXt** | Grouped conv (#1) | **Fully supported** |
| **DeepLab** | Dilated conv (#1) | **Fully supported** |
| **LLaMA** | RMSNorm (#3) + GQA (#8) + RoPE (#9) + SwiGLU | **Mostly supported** (SwiGLU is the remaining blocker) |
| **ViT** | Patch embed (Conv2d) + CLS token + learned PE | **Nearly supported** |
| **U-Net** | ConvTranspose2d (#2) + concat skips | **Partially supported** (concat skips still needed) |
| **Stable Diffusion** | GroupNorm (#5) + cross-attention + concat + time embed | **Partially supported** |
| **EfficientNet** | Grouped conv (#1) + SE blocks | **Partially supported** (SE blocks still needed) |
