from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _pairwise_disagreement_matrix(member_pred_classes: np.ndarray) -> np.ndarray:
    pred_classes = np.asarray(member_pred_classes)
    if pred_classes.ndim != 2:
        raise ValueError(
            "member_pred_classes must have shape (num_members, num_samples)"
        )

    num_members, num_samples = pred_classes.shape
    if num_members < 2:
        raise ValueError("At least two members are required")
    if num_samples == 0:
        raise ValueError("No samples available to compute disagreement")

    disagreement = np.zeros((num_members, num_members), dtype=np.float32)
    for i in range(num_members):
        for j in range(i, num_members):
            rate = float(np.mean(pred_classes[i] != pred_classes[j]))
            disagreement[i, j] = rate
            disagreement[j, i] = rate

    return disagreement


def save_pairwise_disagreement_heatmap(
    member_pred_classes: np.ndarray,
    output_path: str | Path,
    title: str = "Pairwise Disagreement",
) -> np.ndarray:
    disagreement_matrix = _pairwise_disagreement_matrix(member_pred_classes)
    num_members = disagreement_matrix.shape[0]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig_width = max(5.0, 1.2 * num_members)
    fig_height = max(4.5, 1.0 * num_members)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(disagreement_matrix, cmap="magma", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Disagreement rate", rotation=90)

    indices = np.arange(num_members)
    labels = [str(idx) for idx in indices]
    ax.set_title(title)
    ax.set_xlabel("Member index")
    ax.set_ylabel("Member index")
    ax.set_xticks(indices, labels=labels)
    ax.set_yticks(indices, labels=labels)

    for i in range(num_members):
        for j in range(num_members):
            color = "white" if disagreement_matrix[i, j] >= 0.5 else "black"
            ax.text(
                j,
                i,
                f"{disagreement_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                color=color,
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    return disagreement_matrix
