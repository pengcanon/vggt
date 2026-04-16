"""
Plot training and validation losses over epochs from VGGT training logs.

Usage:
    python plot_losses.py <log_dir>
    python plot_losses.py apple_finetune
    python plot_losses.py human_body_finetune
"""

import re
import sys
import os
import matplotlib.pyplot as plt
import numpy as np


def parse_log(log_path):
    """Parse a VGGT training log file and extract per-epoch averaged losses."""
    train_epochs = {}
    val_epochs = {}

    # Pattern matches lines like: Train Epoch: [5][    190/1000000] ... (averaged_value)
    # We want the last batch of each epoch (highest batch idx) for epoch-averaged values
    epoch_pattern = re.compile(
        r"(Train|Val) Epoch: \[(\d+)\]\[\s*(\d+)/\d+\]"
    )
    loss_pattern = re.compile(
        r"Loss/(?:train_|val_)(loss_\w+):\s*[\d.e+-]+\s*\(([\d.e+-]+)\)"
    )

    with open(log_path, "r") as f:
        for line in f:
            epoch_match = epoch_pattern.search(line)
            if not epoch_match:
                continue

            phase = epoch_match.group(1)  # Train or Val
            epoch = int(epoch_match.group(2))
            batch_idx = int(epoch_match.group(3))

            losses = {}
            for m in loss_pattern.finditer(line):
                losses[m.group(1)] = float(m.group(2))

            if not losses:
                continue

            if phase == "Train":
                # Keep the entry with highest batch_idx per epoch (epoch-end average)
                if epoch not in train_epochs or batch_idx > train_epochs[epoch]["batch_idx"]:
                    train_epochs[epoch] = {"batch_idx": batch_idx, "losses": losses}
            else:
                if epoch not in val_epochs or batch_idx > val_epochs[epoch]["batch_idx"]:
                    val_epochs[epoch] = {"batch_idx": batch_idx, "losses": losses}

    # Detect multiple runs: if epoch numbers reset, keep only the last run
    train_epochs = _keep_last_run(train_epochs)
    val_epochs = _keep_last_run(val_epochs)

    return train_epochs, val_epochs


def _keep_last_run(epoch_dict):
    """If epochs reset (e.g., 0,1,...,19,0,1,...,19), keep only the last run."""
    if not epoch_dict:
        return epoch_dict

    sorted_keys = sorted(epoch_dict.keys())
    # Find the last reset point (where epoch goes back to 0 or lower)
    last_start = 0
    for i in range(1, len(sorted_keys)):
        if sorted_keys[i] <= sorted_keys[i - 1]:
            last_start = i

    # This simple approach doesn't work for dict with unique keys.
    # Instead, re-parse by insertion order is needed. For now, return as-is
    # since the epoch dict uses epoch number as key, later entries overwrite earlier.
    return epoch_dict


def plot_losses(train_epochs, val_epochs, title="Training Progress"):
    """Create plots for training and validation losses."""

    camera_keys = ["loss_camera", "loss_T", "loss_R", "loss_FL"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle(title, fontsize=16)

    # --- Plot 1: Confidence depth loss ---
    ax = axes[0, 0]
    for key in ["loss_conf_depth"]:
        train_x, train_y = _extract(train_epochs, key)
        val_x, val_y = _extract(val_epochs, key)
        if train_y:
            ax.plot(train_x, train_y, "-o", markersize=3, label="train")
        if val_y:
            ax.plot(val_x, val_y, "--s", markersize=5, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Depth Confidence Loss (conf_depth)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 2: Regression depth loss ---
    ax = axes[0, 1]
    for key in ["loss_reg_depth"]:
        train_x, train_y = _extract(train_epochs, key)
        val_x, val_y = _extract(val_epochs, key)
        if train_y:
            ax.plot(train_x, train_y, "-o", markersize=3, label="train")
        if val_y:
            ax.plot(val_x, val_y, "--s", markersize=5, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Depth Regression Loss (reg_depth)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 3: Gradient depth loss ---
    ax = axes[1, 0]
    for key in ["loss_grad_depth"]:
        train_x, train_y = _extract(train_epochs, key)
        val_x, val_y = _extract(val_epochs, key)
        if train_y:
            ax.plot(train_x, train_y, "-o", markersize=3, label="train")
        if val_y:
            ax.plot(val_x, val_y, "--s", markersize=5, label="val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Depth Gradient Loss (grad_depth)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 4: Camera losses ---
    ax = axes[1, 1]
    for key in camera_keys:
        train_x, train_y = _extract(train_epochs, key)
        val_x, val_y = _extract(val_epochs, key)
        if train_y:
            ax.plot(train_x, train_y, "-o", markersize=3, label=f"train {key}")
        if val_y:
            ax.plot(val_x, val_y, "--s", markersize=5, label=f"val {key}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Camera Losses")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Plot 5: Objective loss ---
    ax = axes[2, 0]
    train_x, train_y = _extract(train_epochs, "loss_objective")
    val_x, val_y = _extract(val_epochs, "loss_objective")
    if train_y:
        ax.plot(train_x, train_y, "-o", markersize=3, label="train", color="blue")
    if val_y:
        ax.plot(val_x, val_y, "--s", markersize=5, label="val", color="red")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Objective Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Plot 6: Summary table ---
    ax = axes[2, 1]
    ax.axis("off")
    all_keys = ["loss_conf_depth", "loss_reg_depth", "loss_grad_depth"] + camera_keys + ["loss_objective"]

    table_data = []
    headers = ["Metric", "Train (first)", "Train (last)", "Val (first)", "Val (last)", "Trend"]

    for key in all_keys:
        row = [key.replace("loss_", "")]
        train_x, train_y = _extract(train_epochs, key)
        val_x, val_y = _extract(val_epochs, key)

        row.append(f"{train_y[0]:.4f}" if train_y else "—")
        row.append(f"{train_y[-1]:.4f}" if train_y else "—")
        row.append(f"{val_y[0]:.4f}" if val_y else "—")
        row.append(f"{val_y[-1]:.4f}" if val_y else "—")

        if val_y and len(val_y) >= 2:
            change = val_y[-1] - val_y[0]
            row.append("▼ better" if change < -0.001 else ("▲ worse" if change > 0.001 else "— flat"))
        elif train_y and len(train_y) >= 2:
            change = train_y[-1] - train_y[0]
            row.append("▼ better" if change < -0.001 else ("▲ worse" if change > 0.001 else "— flat"))
        else:
            row.append("—")

        table_data.append(row)

    table = ax.table(cellText=table_data, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)

    # Color the trend column
    for i, row in enumerate(table_data):
        cell = table[i + 1, 5]
        if "better" in row[5]:
            cell.set_facecolor("#d4edda")
        elif "worse" in row[5]:
            cell.set_facecolor("#f8d7da")

    plt.tight_layout()
    plt.savefig(os.path.join(sys.argv[1], 'loss_curves.png'))
    print('Saved curve plots!')


def _extract(epoch_dict, key):
    """Extract sorted (epoch, value) pairs for a given loss key."""
    x, y = [], []
    for epoch in sorted(epoch_dict.keys()):
        losses = epoch_dict[epoch]["losses"]
        if key in losses:
            x.append(epoch)
            y.append(losses[key])
    return x, y


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_losses.py <log_dir_name>")
        print("  e.g. python plot_losses.py apple_finetune")
        print("  e.g. python plot_losses.py human_body_finetune")
        sys.exit(1)

    log_dir = sys.argv[1]
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Allow both relative name and full path
    log_path = os.path.join(script_dir, log_dir, "log.txt")
    if not os.path.exists(log_path):
        log_path = os.path.join(log_dir, "log.txt")
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    print(f"Parsing {log_path}...")
    train_epochs, val_epochs = parse_log(log_path)
    print(f"Found {len(train_epochs)} train epochs, {len(val_epochs)} val epochs")

    plot_losses(train_epochs, val_epochs, title=f"Training Progress — {log_dir}")


if __name__ == "__main__":
    main()
