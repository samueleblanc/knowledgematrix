import torch
from torch import nn
from typing import List, Optional, Tuple, Union

from knowledgematrix.neural_net import NN


class KnowledgeMatrixEvolution:
    """
        Closed-form evolution of the Knowledge Matrix under one optimizer step
        for ReLU MLPs, following the decomposition

            KM(theta_{t+1}, x) = KM(theta_t, x) + dKM_smooth + dKM_cross

        where dKM_smooth is the first-order change within the current activation
        polytope and dKM_cross is the (discontinuous) jump produced by any ReLU
        pre-activations flipping sign.

        This first release supports ReLU MLPs only: Flatten/Linear/ReLU layers,
        with the final layer a Linear (no activation after the output). All
        three optimizers from the derivation are supported: vanilla SGD,
        momentum SGD (heavy-ball), and Adam. They share the same decomposition
        and differ only in the per-layer weight update formula.

        Args:
            model (NN): The ReLU MLP whose KM evolution is being tracked.
            optimizer (str): One of "gd", "momentum", "adam".
            lr (float): Learning rate.
            mu (float): Momentum coefficient (momentum optimizer only).
            beta1 (float): Adam first-moment decay (Adam only).
            beta2 (float): Adam second-moment decay (Adam only).
            eps (float): Adam numerical stability constant (Adam only).
            device (Union[str, None]): Compute device; defaults to model.device.
    """

    _ALLOWED_LAYERS = (nn.Flatten, nn.Linear, nn.ReLU)
    _OPTIMIZERS = ("gd", "momentum", "adam")

    def __init__(
            self,
            model: NN,
            optimizer: str = "gd",
            lr: float = 1e-2,
            mu: float = 0.9,
            beta1: float = 0.9,
            beta2: float = 0.999,
            eps: float = 1e-8,
            device: Optional[str] = None,
        ) -> None:
        self._validate_architecture(model)
        if optimizer not in self._OPTIMIZERS:
            raise ValueError(
                f"optimizer must be one of {self._OPTIMIZERS}, got {optimizer!r}"
            )

        self.model = model
        self.optimizer = optimizer
        self.lr = lr
        self.mu = mu
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.device = device if device is not None else model.device

        self._linear_indices: List[int] = [
            i for i, l in enumerate(model.layers) if isinstance(l, nn.Linear)
        ]
        self._relu_indices: List[int] = [
            i for i, l in enumerate(model.layers) if isinstance(l, nn.ReLU)
        ]
        self.num_linear = len(self._linear_indices)

        # Optimizer state, one entry per Linear layer (indexed 0..L-1).
        self._velocity_W: List[Optional[torch.Tensor]] = [None] * self.num_linear
        self._velocity_b: List[Optional[torch.Tensor]] = [None] * self.num_linear
        self._m_W: List[Optional[torch.Tensor]] = [None] * self.num_linear
        self._m_b: List[Optional[torch.Tensor]] = [None] * self.num_linear
        self._s_W: List[Optional[torch.Tensor]] = [None] * self.num_linear
        self._s_b: List[Optional[torch.Tensor]] = [None] * self.num_linear
        self._t: int = 0

        # Populated by forward(), consumable by apply_step().
        self.last_dW: List[Optional[torch.Tensor]] = []
        self.last_db: List[Optional[torch.Tensor]] = []


    ### Public API ###

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Compute dKM_smooth and dKM_cross for one optimizer step at input x.

            Requires: loss.backward() has populated .grad on every Linear layer.
            Side effect: advances the internal optimizer state (velocity for
            momentum, m/s/step counter for Adam). Caches last_dW and last_db
            so that apply_step() can mutate the live model parameters by the
            same updates that produced the returned deltas.

            Args:
                x (torch.Tensor): Input tensor; shape (C,H,W) or (1,C,H,W).
            Returns:
                Tuple[torch.Tensor, torch.Tensor]: (dKM_smooth, dKM_cross),
                both of shape (C_out, d+1).
        """
        with torch.no_grad():
            x_flat = x.reshape(-1).to(self.device)
            weights, biases = self._snapshot_params()
            masks_old = self._collect_masks(x)

            dW, db = self._compute_deltas()
            self.last_dW = dW
            self.last_db = db

            # The decomposition used is
            #     KM(theta_{t+1}) = KM(theta_t) + dKM_cross + dKM_smooth + O(lr^2)
            # with dKM_cross built at the OLD parameters and dKM_smooth built
            # at the NEW masks D_{t+1}. This is the form stated in the
            # "separating smooth and crossing terms" remark of the tex and
            # yields an O(lr^2) residual even when activations flip (using
            # old masks would instead give an O(lr) residual at flipped
            # coordinates).
            masks_new = self._forward_with_shifted_params(x_flat, weights, biases, dW, db)
            delta_cross = self._cross_term(x_flat, weights, biases, masks_old, masks_new)
            delta_smooth = self._smooth_term(x_flat, weights, biases, masks_new, dW, db)

        return delta_smooth, delta_cross

    def apply_step(self) -> None:
        """
            Mutate the live model parameters by the last-computed updates.
            Equivalent to running the corresponding optimizer step externally,
            but guaranteed to use exactly the same dW/db that produced the
            last returned (dKM_smooth, dKM_cross).
        """
        if not self.last_dW:
            raise RuntimeError("Call forward() before apply_step().")
        with torch.no_grad():
            for idx, lin_idx in enumerate(self._linear_indices):
                layer = self.model.layers[lin_idx]
                layer.weight.add_(self.last_dW[idx].to(layer.weight.device))
                if layer.bias is not None and self.last_db[idx] is not None:
                    layer.bias.add_(self.last_db[idx].to(layer.bias.device))


    ### Validation ###

    def _validate_architecture(self, model: NN) -> None:
        """Only Flatten/Linear/ReLU allowed; last layer must be Linear."""
        bad = [
            type(l).__name__
            for l in model.layers
            if not isinstance(l, self._ALLOWED_LAYERS)
        ]
        if bad:
            raise ValueError(
                f"KnowledgeMatrixEvolution only supports Flatten/Linear/ReLU "
                f"MLPs; found unsupported layer(s): {bad}"
            )
        if not any(isinstance(l, nn.Linear) for l in model.layers):
            raise ValueError("Model has no Linear layers.")
        if not isinstance(model.layers[-1], nn.Linear):
            raise ValueError("Last layer must be Linear (no activation after output).")


    ### Parameter snapshots and masks ###

    def _snapshot_params(self) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        weights: List[torch.Tensor] = []
        biases: List[Optional[torch.Tensor]] = []
        for idx in self._linear_indices:
            layer = self.model.layers[idx]
            weights.append(layer.weight.detach().to(self.device))
            biases.append(
                layer.bias.detach().to(self.device) if layer.bias is not None else None
            )
        return weights, biases

    def _collect_masks(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
            Run model.forward(x) with save=True and extract binary masks from
            the cached pre-activations at each ReLU layer. One vector per
            ReLU, of shape (n_l,).
        """
        old_save = self.model.save
        self.model.save = True
        _ = self.model.forward(x)
        self.model.save = old_save
        masks: List[torch.Tensor] = []
        dtype = next(self.model.parameters()).dtype
        for relu_idx in self._relu_indices:
            pre = self.model.pre_acts[relu_idx]
            masks.append((pre > 0).to(dtype=dtype, device=self.device).reshape(-1))
        return masks


    ### Optimizer deltas ###

    def _compute_deltas(
        self,
    ) -> Tuple[List[torch.Tensor], List[Optional[torch.Tensor]]]:
        L = self.num_linear
        dW: List[Optional[torch.Tensor]] = [None] * L
        db: List[Optional[torch.Tensor]] = [None] * L

        grads_W: List[torch.Tensor] = []
        grads_b: List[Optional[torch.Tensor]] = []
        for idx in self._linear_indices:
            layer = self.model.layers[idx]
            if layer.weight.grad is None:
                raise RuntimeError(
                    "Linear layer has no .grad; call loss.backward() before forward()."
                )
            grads_W.append(layer.weight.grad.detach().to(self.device))
            if layer.bias is not None:
                if layer.bias.grad is None:
                    raise RuntimeError(
                        "Linear layer bias has no .grad; call loss.backward() first."
                    )
                grads_b.append(layer.bias.grad.detach().to(self.device))
            else:
                grads_b.append(None)

        if self.optimizer == "gd":
            for i in range(L):
                dW[i] = -self.lr * grads_W[i]
                db[i] = -self.lr * grads_b[i] if grads_b[i] is not None else None

        elif self.optimizer == "momentum":
            for i in range(L):
                if self._velocity_W[i] is None:
                    self._velocity_W[i] = torch.zeros_like(grads_W[i])
                self._velocity_W[i] = self.mu * self._velocity_W[i] + grads_W[i]
                dW[i] = -self.lr * self._velocity_W[i]
                if grads_b[i] is not None:
                    if self._velocity_b[i] is None:
                        self._velocity_b[i] = torch.zeros_like(grads_b[i])
                    self._velocity_b[i] = self.mu * self._velocity_b[i] + grads_b[i]
                    db[i] = -self.lr * self._velocity_b[i]

        elif self.optimizer == "adam":
            self._t += 1
            b1t = 1.0 - self.beta1 ** self._t
            b2t = 1.0 - self.beta2 ** self._t
            for i in range(L):
                if self._m_W[i] is None:
                    self._m_W[i] = torch.zeros_like(grads_W[i])
                    self._s_W[i] = torch.zeros_like(grads_W[i])
                self._m_W[i] = self.beta1 * self._m_W[i] + (1 - self.beta1) * grads_W[i]
                self._s_W[i] = self.beta2 * self._s_W[i] + (1 - self.beta2) * grads_W[i] ** 2
                mhat = self._m_W[i] / b1t
                shat = self._s_W[i] / b2t
                dW[i] = -self.lr * mhat / (torch.sqrt(shat) + self.eps)
                if grads_b[i] is not None:
                    if self._m_b[i] is None:
                        self._m_b[i] = torch.zeros_like(grads_b[i])
                        self._s_b[i] = torch.zeros_like(grads_b[i])
                    self._m_b[i] = self.beta1 * self._m_b[i] + (1 - self.beta1) * grads_b[i]
                    self._s_b[i] = self.beta2 * self._s_b[i] + (1 - self.beta2) * grads_b[i] ** 2
                    mhatb = self._m_b[i] / b1t
                    shatb = self._s_b[i] / b2t
                    db[i] = -self.lr * mhatb / (torch.sqrt(shatb) + self.eps)

        return dW, db


    ### Propagators and KM assembly ###

    def _propagators(
        self,
        weights: List[torch.Tensor],
        masks: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
            Build upper propagators U[i] and lower propagators Low[i] with
            0-indexed convention (i = l - 1 in the tex), for l = 1..L.

                U[L-1] = I_C
                U[i]   = U[i+1] @ W[i+1] @ diag(masks[i])        (i = L-2..0)

                Low[0] = I_d
                Low[i] = diag(masks[i-1]) @ W[i-1] @ Low[i-1]    (i = 1..L-1)
        """
        L = self.num_linear
        C = weights[-1].shape[0]
        d = weights[0].shape[1]
        dtype = weights[0].dtype

        U: List[Optional[torch.Tensor]] = [None] * L
        U[L - 1] = torch.eye(C, device=self.device, dtype=dtype)
        for i in range(L - 2, -1, -1):
            U[i] = U[i + 1] @ weights[i + 1] @ torch.diag(masks[i])

        Low: List[Optional[torch.Tensor]] = [None] * L
        Low[0] = torch.eye(d, device=self.device, dtype=dtype)
        for i in range(1, L):
            Low[i] = torch.diag(masks[i - 1]) @ weights[i - 1] @ Low[i - 1]

        return U, Low  # type: ignore[return-value]

    def _bias_recursion(
        self,
        weights: List[torch.Tensor],
        biases: List[Optional[torch.Tensor]],
        masks: List[torch.Tensor],
    ) -> List[torch.Tensor]:
        """
            cr[i] = c^{(i+1)} following the recursion
                c^{(1)} = b^{(1)}
                c^{(l)} = W^{(l)} * diag(D^{(l-1)}) * c^{(l-1)} + b^{(l)}
            with the bias treated as zero when a layer has no bias.
        """
        L = self.num_linear
        dtype = weights[0].dtype
        cr: List[torch.Tensor] = []
        b0 = biases[0] if biases[0] is not None else torch.zeros(
            weights[0].shape[0], device=self.device, dtype=dtype
        )
        cr.append(b0)
        for i in range(1, L):
            masked_prev = masks[i - 1] * cr[i - 1]
            out = weights[i] @ masked_prev
            if biases[i] is not None:
                out = out + biases[i]
            cr.append(out)
        return cr

    def _km_with_masks(
        self,
        x_flat: torch.Tensor,
        weights: List[torch.Tensor],
        biases: List[Optional[torch.Tensor]],
        masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """KM(theta, x) evaluated with an externally supplied mask list."""
        L = self.num_linear
        C = weights[-1].shape[0]
        dtype = weights[0].dtype

        J = weights[0]
        for i in range(1, L):
            J = weights[i] @ torch.diag(masks[i - 1]) @ J

        cr = self._bias_recursion(weights, biases, masks)
        c = cr[-1]
        if c is None:
            c = torch.zeros(C, device=self.device, dtype=dtype)

        return torch.cat([J * x_flat.unsqueeze(0), c.unsqueeze(1)], dim=1)


    ### Smooth and crossing terms ###

    def _smooth_term(
        self,
        x_flat: torch.Tensor,
        weights: List[torch.Tensor],
        biases: List[Optional[torch.Tensor]],
        masks: List[torch.Tensor],
        dW: List[torch.Tensor],
        db: List[Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """
            dKM_smooth = [dJ * diag(x) | dc]
                dJ = sum_l U^{(l)} dW^{(l)} L^{(l)}
                dc = sum_l U^{(l)} ( dW^{(l)} D^{(l-1)} c^{(l-1)} + db^{(l)} )
        """
        L = self.num_linear
        d = x_flat.shape[0]
        C = weights[-1].shape[0]
        dtype = weights[0].dtype

        U, Low = self._propagators(weights, masks)
        cr = self._bias_recursion(weights, biases, masks)

        delta_J = torch.zeros((C, d), device=self.device, dtype=dtype)
        for i in range(L):
            delta_J = delta_J + U[i] @ dW[i] @ Low[i]

        delta_c = torch.zeros(C, device=self.device, dtype=dtype)
        for i in range(L):
            # D^{(l-1)} c^{(l-1)} with l = i+1: for i=0 the bias recursion
            # start uses c^{(0)} = 0, so that term vanishes.
            if i == 0:
                contrib = torch.zeros(weights[0].shape[0], device=self.device, dtype=dtype)
            else:
                contrib = dW[i] @ (masks[i - 1] * cr[i - 1])
            if db[i] is not None:
                contrib = contrib + db[i]
            delta_c = delta_c + U[i] @ contrib

        return torch.cat([delta_J * x_flat.unsqueeze(0), delta_c.unsqueeze(1)], dim=1)

    def _cross_term(
        self,
        x_flat: torch.Tensor,
        weights: List[torch.Tensor],
        biases: List[Optional[torch.Tensor]],
        masks_old: List[torch.Tensor],
        masks_new: List[torch.Tensor],
    ) -> torch.Tensor:
        """dKM_cross = KM(theta_t, x; D_{t+1}) - KM(theta_t, x; D_t)."""
        km_new = self._km_with_masks(x_flat, weights, biases, masks_new)
        km_old = self._km_with_masks(x_flat, weights, biases, masks_old)
        return km_new - km_old

    def _forward_with_shifted_params(
        self,
        x_flat: torch.Tensor,
        weights: List[torch.Tensor],
        biases: List[Optional[torch.Tensor]],
        dW: List[torch.Tensor],
        db: List[Optional[torch.Tensor]],
    ) -> List[torch.Tensor]:
        """
            Run a forward pass using (W + dW, b + db) without mutating the
            model, and return the resulting ReLU masks per layer.
        """
        L = self.num_linear
        dtype = weights[0].dtype
        h = x_flat.to(device=self.device, dtype=dtype)
        masks: List[torch.Tensor] = []
        for i in range(L):
            W_new = weights[i] + dW[i]
            z = W_new @ h
            if biases[i] is not None:
                b_new = biases[i] if db[i] is None else biases[i] + db[i]
                z = z + b_new
            if i < L - 1:
                mask = (z > 0).to(dtype=dtype, device=self.device)
                masks.append(mask)
                h = mask * z
        return masks
