"""
Exact feature attribution maps and visualization utilities for knowledge matrices.

The knowledge matrix A of shape (output_size, input_size) has the property that
output[j] = sum_i A[j, i]. This means A[j, :] gives the exact contribution of
each input element to output j -- not an approximation like Grad-CAM or SHAP,
but the mathematically exact decomposition of the network's computation.

For image classifiers, reshaping A[j, :] back to (C, H, W) produces an attribution
map showing exactly which pixels drove the prediction for class j. Comparing maps
across classes reveals what the network focuses on differently for each prediction.

Typical workflow:
    model = MyCNN(input_shape=(3, 32, 32), num_classes=10)
    matrix_computer = KnowledgeMatrixComputer(model, batch_size=16)
    A = matrix_computer.forward(x)

    # Get attribution heatmap for class 0
    heatmap = attribution_map(A, output_class=0, input_shape=(3, 32, 32))

    # Overlay on the original image
    overlay_attribution(x, heatmap)

    # Compare attributions across classes
    per_class_comparison(A, classes=[0, 1, 2], input_shape=(3, 32, 32))
"""

import torch
from typing import Union


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def attribution_map(
    A: torch.Tensor,
    output_class: int,
    input_shape: tuple,
    reduce_channels: bool = True
) -> torch.Tensor:
    """
    Extract the attribution map for a given output class from the knowledge matrix.

    Since A[j, i] is the exact contribution of input element i to output j,
    reshaping A[j, :] to (C, H, W) produces a per-pixel attribution map.

    Args:
        A: Knowledge matrix of shape (output_size, input_size) or
           (output_size, input_size + 1) if bias column is present.
        output_class: Index of the output neuron to attribute.
        input_shape: (C, H, W) shape of the network input.
        reduce_channels: If True and C > 1, sum across channels to produce
                         a single (H, W) heatmap.

    Returns:
        Attribution tensor of shape (H, W) if reduce_channels=True,
        or (C, H, W) if reduce_channels=False.
    """
    C, H, W = input_shape
    input_size = C * H * W
    attr = A[output_class, :input_size].reshape(C, H, W)
    if reduce_channels:
        attr = attr.sum(dim=0)
    return attr


def overlay_attribution(
    image: torch.Tensor,
    attribution: torch.Tensor,
    cmap: str = "seismic",
    alpha: float = 0.5,
    symmetric: bool = True,
    show: bool = True
):
    """
    Overlay an attribution heatmap on the original input image.

    Args:
        image: Input image tensor of shape (C, H, W).
        attribution: 2D attribution map of shape (H, W),
                     typically from attribution_map(..., reduce_channels=True).
        cmap: Matplotlib colormap. Default "seismic" (red=positive, blue=negative).
        alpha: Transparency of the heatmap overlay.
        symmetric: If True, center the colormap at 0 so positive/negative
                   contributions are visually balanced.
        show: If True, call plt.show().

    Returns:
        matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    fig, ax = plt.subplots(1, 1)

    img = image.detach().cpu()
    if img.shape[0] == 1:
        ax.imshow(img.squeeze(0), cmap="gray")
    elif img.shape[0] == 3:
        ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
    else:
        ax.imshow(img[0], cmap="gray")

    attr = attribution.detach().cpu()
    vmax = attr.abs().max().item() if symmetric else None
    vmin = -vmax if symmetric and vmax else None
    ax.imshow(attr, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    ax.axis("off")

    if show:
        plt.show()
    return fig


def top_k_contributors(
    A: torch.Tensor,
    output_class: int,
    k: int,
    input_shape: tuple
) -> tuple:
    """
    Identify the k input positions with the largest absolute contribution
    to a given output class.

    Args:
        A: Knowledge matrix of shape (output_size, input_size [+1 bias]).
        output_class: Index of the output neuron.
        k: Number of top contributors to return.
        input_shape: (C, H, W) shape of the network input.

    Returns:
        Tuple of (values, flat_indices, spatial_indices) where:
        - values: Tensor of shape (k,) with signed contribution values,
                  sorted by absolute magnitude descending.
        - flat_indices: Tensor of shape (k,) with flat input positions.
        - spatial_indices: Tensor of shape (k, 3) with (channel, height, width)
                          positions for each contributor.
    """
    C, H, W = input_shape
    input_size = C * H * W
    row = A[output_class, :input_size]
    _, flat_indices = row.abs().topk(k)
    values = row[flat_indices]
    c = flat_indices // (H * W)
    rem = flat_indices % (H * W)
    h = rem // W
    w = rem % W
    spatial_indices = torch.stack([c, h, w], dim=1)
    return values, flat_indices, spatial_indices


def per_class_comparison(
    A: torch.Tensor,
    classes: list,
    input_shape: tuple,
    cmap: str = "seismic",
    symmetric: bool = True,
    class_names: Union[list, None] = None,
    show: bool = True
):
    """
    Show side-by-side attribution maps for multiple output classes.

    Uses a shared color scale across all classes so contributions are
    visually comparable.

    Args:
        A: Knowledge matrix.
        classes: List of output class indices to compare.
        input_shape: (C, H, W) shape of the network input.
        cmap: Matplotlib colormap.
        symmetric: If True, center the colormap at 0.
        class_names: Optional list of labels for each class.
        show: If True, call plt.show().

    Returns:
        matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    n = len(classes)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    attrs = [attribution_map(A, cls, input_shape).detach().cpu() for cls in classes]
    global_vmax = max(a.abs().max().item() for a in attrs) if symmetric else None
    global_vmin = -global_vmax if symmetric and global_vmax else None

    for i, attr in enumerate(attrs):
        axes[i].imshow(attr, cmap=cmap, vmin=global_vmin, vmax=global_vmax)
        title = class_names[i] if class_names else f"Class {classes[i]}"
        axes[i].set_title(title)
        axes[i].axis("off")

    plt.tight_layout()
    if show:
        plt.show()
    return fig


def layer_contribution(
    A_layers: Union[dict, list],
    show: bool = True
):
    """
    Bar chart of per-layer contribution magnitude (Frobenius norm).

    Args:
        A_layers: Either a dict mapping layer names to contribution tensors,
                  or a list of contribution tensors (auto-named "Layer 0", etc.).
        show: If True, call plt.show().

    Returns:
        matplotlib.figure.Figure
    """
    plt = _require_matplotlib()
    if isinstance(A_layers, list):
        names = [f"Layer {i}" for i in range(len(A_layers))]
        matrices = A_layers
    else:
        names = list(A_layers.keys())
        matrices = list(A_layers.values())

    magnitudes = [m.norm().item() for m in matrices]

    fig, ax = plt.subplots(1, 1, figsize=(max(6, len(names)), 4))
    ax.bar(range(len(names)), magnitudes)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Contribution magnitude (Frobenius norm)")
    ax.set_title("Per-layer contribution")
    plt.tight_layout()
    if show:
        plt.show()
    return fig
