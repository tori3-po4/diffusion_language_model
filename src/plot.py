"""Plot training metrics from CSV log. Supports live monitoring during training."""

import argparse
import csv
import time

import os
import matplotlib
if os.environ.get("COLAB_RELEASE_TAG"):
    matplotlib.use("module://matplotlib_inline.backend_inline")
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def load_metrics(csv_path: str):
    steps, losses, lrs = [], [], []
    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
                lrs.append(float(row["lr"]))
    except FileNotFoundError:
        pass
    return steps, losses, lrs


def smooth(values, window):
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out.append(sum(values[start : i + 1]) / (i - start + 1))
    return out


def plot_once(csv_path: str, output_path: str, window: int = 10):
    """Static plot (save to file)."""
    steps, losses, lrs = load_metrics(csv_path)
    if not steps:
        print("No data to plot.")
        return

    smoothed = smooth(losses, window)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(steps, losses, alpha=0.3, color="C0", label="Raw loss")
    ax1.plot(steps, smoothed, color="C0", linewidth=2, label=f"Smoothed (window={window})")
    ax1.set_ylabel("Loss")
    ax1.set_title("MDLM Training")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps, lrs, color="C1", linewidth=2)
    ax2.set_ylabel("Learning Rate")
    ax2.set_xlabel("Step")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {output_path}")

    plt.show()


def plot_live(csv_path: str, window: int = 10, refresh: float = 5.0):
    """Live monitoring: auto-refresh plot while training runs."""
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("MDLM Training (live)")

    prev_len = 0
    print(f"Watching {csv_path} (refresh every {refresh}s, close window to stop)")

    while True:
        steps, losses, lrs = load_metrics(csv_path)

        if len(steps) > prev_len:
            prev_len = len(steps)
            smoothed = smooth(losses, window)

            ax1.clear()
            ax1.plot(steps, losses, alpha=0.3, color="C0", label="Raw loss")
            ax1.plot(steps, smoothed, color="C0", linewidth=2,
                     label=f"Smoothed (window={window})")
            ax1.set_ylabel("Loss")
            ax1.legend(loc="upper right")
            ax1.grid(True, alpha=0.3)

            ax2.clear()
            ax2.plot(steps, lrs, color="C1", linewidth=2)
            ax2.set_ylabel("Learning Rate")
            ax2.set_xlabel("Step")
            ax2.grid(True, alpha=0.3)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        if not plt.fignum_exists(fig.number):
            break

        try:
            plt.pause(refresh)
        except Exception:
            break


def main():
    parser = argparse.ArgumentParser(description="Plot MDLM training metrics")
    parser.add_argument("--csv", type=str, default="logs/metrics.csv")
    parser.add_argument("--output", type=str, default="logs/training_curve.png")
    parser.add_argument("--window", type=int, default=10, help="Smoothing window size")
    parser.add_argument("--live", action="store_true",
                        help="Live monitoring mode: auto-refresh during training")
    parser.add_argument("--refresh", type=float, default=5.0,
                        help="Refresh interval in seconds (live mode)")
    args = parser.parse_args()

    if args.live:
        plot_live(args.csv, args.window, args.refresh)
    else:
        plot_once(args.csv, args.output, args.window)


if __name__ == "__main__":
    main()
