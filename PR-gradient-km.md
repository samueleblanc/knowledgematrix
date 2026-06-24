# Add Gradient-Based Knowledge Matrix Computer (Exact-or-Raise for Piecewise-Linear Nets)

## Summary

This PR adds a new autograd-based knowledge-matrix computer, `GradientMatrixComputer`, as an alternative to the existing secant-based `KnowledgeMatrixComputer`.

- **The formulation.** The knowledge matrix is `M(θ, x) = [ J(θ, x)·diag(x) | c(θ, x) ]` of shape `(C, d+1)`:
  - columns `1..d` are per-class **gradient × input**: `M[c, i] = (∂f_c/∂x_i)·x_i`
  - column `d+1` is the **FullGrad bias attribution**: `c_c = Σ_b (∂f_c/∂b)·b_eff`
  - the row-sum identity `M·1 = f(x)` (the KM invariant) holds exactly on the supported model class.
- **PL-only exactness (the key result).** This gradient×input + bias decomposition is exact **if and only if the network is piecewise-linear** (ReLU-family activations, affine layers, max/avg pooling). For PL nets the function is locally affine, so `f_c(x) = Σ_i J[c, i]·x_i + c_c` holds with no remainder. For smooth activations (GELU/sigmoid/tanh) the pointwise gradient×input + bias carries a Taylor remainder and does **not** sum to the output. The existing `KnowledgeMatrixComputer` stays exact on smooth nets only because it uses the **secant** slope `post_act/pre_act`, not the gradient. So the gradient formulation is a piecewise-linear phenomenon.
- **Exact-or-raise contract.** Because the formulation is only exact on PL nets, the computer is **exact-or-raise**: it walks the model's layers and raises `ValueError` on the first non-PL layer rather than returning a silently-wrong matrix.
- **FullGrad bias (not `f − J·x`).** The bias column is computed independently of `f(x)` via a single forward-mode `torch.func.jvp` in the direction of the effective additive biases. This makes the row-sum `J·x + c = f` a **genuine Euler-identity check** (holds for PL, fails for smooth) rather than tautological. Effective bias: raw `.bias` for `Linear`/`Conv`/`ConvTranspose`; `β − γ·μ/√(var+eps)` for `BatchNorm2d` in eval mode.

## New API

New module `knowledgematrix/gradient_matrix.py`, class `GradientMatrixComputer`:

```python
GradientMatrixComputer(model, batch_size=1, device=None, backend="func")
```

- `forward(x, extract_weff=False)` returns the `(C, d+1)` matrix (gradient×input columns + bias column), or the bare input Jacobian `J` of shape `(C, d)` when `extract_weff=True`.
- `backend="func"` (default) uses `torch.func.jacrev` for the Jacobian (C backward passes); `backend="autograd"` uses a chunked `torch.autograd.grad` loop (`batch_size` classes per chunk). An invalid backend is rejected.
- The PL guard raises `ValueError` on the first non-PL layer; `forward` raises `RuntimeError` if the model's layers are not in eval mode.
- No new dependencies (`torch.func` ships with torch ≥ 2.0). The root `__init__.py` is unchanged — import by full path.

## Files Changed

| File | Change |
|------|--------|
| `knowledgematrix/gradient_matrix.py` | New: `GradientMatrixComputer` (PL guard, `func`/`autograd` Jacobian backends, FullGrad bias column via forward-mode `jvp`, `extract_weff` returning the bare input Jacobian, eval-mode enforcement) |
| `extra/tests/gradient_matrix.py` | New: `unittest` suite (float64, 12 tests) covering the PL guard, Jacobian/KM cross-validation against the reference computer, the KM invariant, backend parity, and a printed speed comparison |

## Performance

The gradient method does C backward passes vs the reference's `d+1` forward passes, so the speedup grows with input dimension `d`. Measured on CPU/float64 (mean of 3), from `extra/tests/gradient_matrix.py::TestSpeed`:

```
--- KM speed comparison (per call, mean of 3) ---
network             d    C     KM (ms)   grad (ms)   speedup
MLP-28x28         784   10        3.80        1.33      2.8x
MLP-64x64        4096   10       64.06        2.04     31.4x
CNN-3x16x16       768   10        6.89        2.11      3.3x
```

## Test Results

`extra/tests/gradient_matrix.py` — `unittest`, float64, **12 tests, all passing**. Run with:

```bash
python extra/tests/gradient_matrix.py
```

Coverage:

- **PL guard** — accepts PL models, raises `ValueError` on a smooth (GELU) net, rejects an invalid backend.
- **Input Jacobian** — correct `(C, d)` shape; machine-epsilon agreement with `KnowledgeMatrixComputer` `extract_weff`.
- **KM invariant** — `mat.sum(1) == forward(x)` on CNN / MLP / BatchNorm models.
- **Full-matrix agreement** — matches the reference computer including BatchNorm with non-trivial running stats, validating the `β − γμ/σ` effective-bias formula at ~`1e-18`.
- **Backend parity** — `func` vs `autograd` produce identical matrices.
- **Robustness** — zero-pixel inputs stay finite; output shapes correct; eval-mode enforcement raises `RuntimeError` when layers are in train mode.
- **Speed** — prints the comparison table above.
