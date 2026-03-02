"""MDLM training loop with HuggingFace Accelerate."""

import os
import sys
import csv
import copy
import math
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
from torch.optim import AdamW
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

from model import DIT, subs_parameterization
from noise_schedule import LogLinearNoise, q_xt
from dataset import get_dataloader


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps):
    """Cosine LR schedule with linear warmup."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_loss(
    model: DIT,
    x0: torch.Tensor,
    noise_schedule: LogLinearNoise,
    mask_token_id: int,
    antithetic: bool = True,
):
    """Compute MDLM ELBO loss for a batch.

    1. Sample t ~ Uniform(eps, 1)  (with antithetic sampling for variance reduction)
    2. Compute sigma(t) and move_chance(t)
    3. Mask tokens: xt = q_xt(x0, move_chance)
    4. Forward pass: model(xt, sigma) -> logits
    5. SUBS parameterization -> log_probs
    6. Weighted cross-entropy loss: weight = dsigma/dt / expm1(sigma)
    """
    B = x0.shape[0]
    device = x0.device
    eps = noise_schedule.eps

    # Sample t with antithetic sampling
    if antithetic:
        half_B = B // 2
        t = torch.rand(half_B, device=device) * (1.0 - eps) + eps
        t = torch.cat([t, 1.0 - t + eps], dim=0)[:B]
    else:
        t = torch.rand(B, device=device) * (1.0 - eps) + eps

    t = t.clamp(eps, 1.0)

    sigma = noise_schedule.sigma(t)
    move_chance = noise_schedule.move_chance(t)

    # Forward diffusion
    xt = q_xt(x0, move_chance[:, None], mask_token_id)

    # Model forward
    logits = model(xt, sigma)

    # SUBS parameterization: NLL at masked positions only (memory-efficient)
    nll = subs_parameterization(logits, xt, x0, mask_token_id)  # (B, S)

    # Loss weight: dsigma/dt / (exp(sigma) - 1)
    rate = noise_schedule.rate(t)
    expm1_sigma = torch.expm1(sigma)
    loss_weight = rate / expm1_sigma  # (B,)

    nll_per_sample = nll.sum(dim=-1)  # (B,)

    # Normalize by sequence length
    seq_len = x0.shape[1]
    loss = (loss_weight * nll_per_sample).mean() / seq_len

    return loss


@torch.no_grad()
def update_ema(ema_model, model, decay):
    """Exponential moving average update."""
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(p.data, alpha=1 - decay)


class MetricsLogger:
    """Log training metrics to CSV file."""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, "metrics.csv")
        self._file = open(self.csv_path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["step", "loss", "lr"])
        self._file.flush()

    def log(self, step: int, loss: float, lr: float):
        self._writer.writerow([step, f"{loss:.6f}", f"{lr:.8f}"])
        self._file.flush()

    def close(self):
        self._file.close()


def main():
    parser = argparse.ArgumentParser(description="MDLM Pre-training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    tc = config["training"]
    dc = config["data"]
    nc = config["noise"]
    wc = config.get("wandb", {})

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=tc["mixed_precision"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        log_with="wandb" if wc.get("enabled") else None,
    )
    set_seed(tc["seed"])

    # Metrics logger (main process only)
    metrics_logger = None
    if accelerator.is_main_process:
        metrics_logger = MetricsLogger("logs")
        if wc.get("enabled"):
            accelerator.init_trackers(
                project_name=wc["project"],
                config=config,
            )

    # Build model
    model = DIT(
        vocab_size=mc["vocab_size"],
        mask_token_id=mc["mask_token_id"],
        hidden_dim=mc["hidden_dim"],
        n_layers=mc["n_layers"],
        n_heads=mc["n_heads"],
        seq_len=mc["seq_len"],
        dropout=mc["dropout"],
        weight_tying=mc["weight_tying"],
    )

    if accelerator.is_main_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # EMA model
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)

    # Noise schedule
    noise_schedule = LogLinearNoise(eps=nc["eps"])
    mask_token_id = mc["mask_token_id"]

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=tc["lr"],
        weight_decay=tc["weight_decay"],
        betas=(0.9, 0.999),
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, tc["warmup_steps"], tc["max_steps"])

    # Dataloader
    dataloader = get_dataloader(
        dataset_name=dc["dataset_name"],
        dataset_config=dc["dataset_config"],
        seq_len=mc["seq_len"],
        batch_size=tc["batch_size"],
        shuffle_buffer=dc["shuffle_buffer"],
        seed=tc["seed"],
    )

    # Prepare with accelerate
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )
    ema_model.to(accelerator.device)

    # Training loop
    global_step = 0
    data_iter = iter(dataloader)

    progress_bar = tqdm(
        range(tc["max_steps"]),
        desc="Training",
        disable=not accelerator.is_main_process,
    )

    model.train()
    for _ in progress_bar:
        # Get batch
        try:
            x0 = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x0 = next(data_iter)

        with accelerator.accumulate(model):
            loss = compute_loss(
                accelerator.unwrap_model(model),
                x0,
                noise_schedule,
                mask_token_id,
            )

            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), tc["max_grad_norm"])

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            global_step += 1
            update_ema(ema_model, accelerator.unwrap_model(model), tc["ema_decay"])

            # Logging
            if global_step % tc["log_every"] == 0:
                lr = scheduler.get_last_lr()[0]
                loss_val = loss.item()

                if accelerator.is_main_process:
                    progress_bar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")
                    metrics_logger.log(global_step, loss_val, lr)
                    if wc.get("enabled"):
                        accelerator.log(
                            {"loss": loss_val, "lr": lr, "step": global_step},
                            step=global_step,
                        )

            # Save checkpoint
            if global_step % tc["save_every"] == 0 and accelerator.is_main_process:
                save_dir = os.path.join("checkpoints", f"step_{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                accelerator.save_state(save_dir)
                torch.save(ema_model.state_dict(), os.path.join(save_dir, "ema_model.pt"))
                print(f"Saved checkpoint at step {global_step}")

        if global_step >= tc["max_steps"]:
            break

    # Final save
    if accelerator.is_main_process:
        save_dir = os.path.join("checkpoints", "final")
        os.makedirs(save_dir, exist_ok=True)
        accelerator.save_state(save_dir)
        torch.save(ema_model.state_dict(), os.path.join(save_dir, "ema_model.pt"))
        metrics_logger.close()
        print(f"Training complete. Metrics saved to {metrics_logger.csv_path}")

    if wc.get("enabled"):
        accelerator.end_training()


if __name__ == "__main__":
    main()
