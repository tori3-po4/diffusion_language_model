"""Iterative denoising sampler for MDLM (DDPM-cache sampling)."""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from model import DIT
from noise_schedule import LogLinearNoise


@torch.no_grad()
def sample(
    model: DIT,
    noise_schedule: LogLinearNoise,
    seq_len: int = 256,
    num_samples: int = 8,
    num_steps: int = 256,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate text by iterative denoising from all-[MASK].

    Uses DDPM-cache sampler: at each step from t to s (t > s),
    - Compute model predictions at time t
    - For currently masked positions, unmask with probability
      (move_chance_t - move_chance_s) / move_chance_t
    - Sample new token from model predictions for positions being unmasked

    Args:
        model: Trained DIT model.
        noise_schedule: Noise schedule instance.
        seq_len: Sequence length.
        num_samples: Number of sequences to generate.
        num_steps: Number of denoising steps.
        device: Device to run on.

    Returns:
        Generated token indices, shape (num_samples, seq_len).
    """
    mask_token_id = model.mask_token_id

    # Start from all [MASK]
    xt = torch.full((num_samples, seq_len), mask_token_id, dtype=torch.long, device=device)

    # Time steps from 1.0 → 0.0 (reverse process)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

    for i in tqdm(range(num_steps), desc="Sampling"):
        t = timesteps[i]
        s = timesteps[i + 1]

        # Current noise level
        sigma_t = noise_schedule.sigma(t.unsqueeze(0).expand(num_samples))
        move_chance_t = noise_schedule.move_chance(t.unsqueeze(0).expand(num_samples))
        move_chance_s = noise_schedule.move_chance(s.unsqueeze(0).expand(num_samples))

        # Model prediction at masked positions
        logits = model(xt, sigma_t)
        # Only sample from real vocab (exclude [MASK] token)
        logits_vocab = logits[:, :, :mask_token_id].float()
        sampled_tokens = torch.distributions.Categorical(logits=logits_vocab).sample()

        # Decide which masked positions to unmask
        is_mask = (xt == mask_token_id)

        if s > 0:
            # Probability of staying masked: move_chance_s / move_chance_t
            # Probability of unmasking: 1 - move_chance_s / move_chance_t
            stay_masked_prob = move_chance_s / move_chance_t  # (B,)
            stay_masked = torch.rand_like(xt.float()) < stay_masked_prob[:, None]
            unmask = is_mask & ~stay_masked
        else:
            # Final step: unmask everything
            unmask = is_mask

        xt = torch.where(unmask, sampled_tokens, xt)

    return xt


def main():
    parser = argparse.ArgumentParser(description="MDLM Text Generation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    mc = config["model"]
    nc = config["noise"]

    # Build model
    model = DIT(
        vocab_size=mc["vocab_size"],
        mask_token_id=mc["mask_token_id"],
        hidden_dim=mc["hidden_dim"],
        n_layers=mc["n_layers"],
        n_heads=mc["n_heads"],
        seq_len=mc["seq_len"],
        dropout=0.0,
        weight_tying=mc["weight_tying"],
    )

    # Load weights
    if args.use_ema:
        ema_path = f"{args.checkpoint}/ema_model.pt"
        state_dict = torch.load(ema_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded EMA model from {ema_path}")
    else:
        from accelerate import load_checkpoint_in_model
        load_checkpoint_in_model(model, args.checkpoint)
        print(f"Loaded model from {args.checkpoint}")

    model.to(args.device)
    model.eval()

    noise_schedule = LogLinearNoise(eps=nc["eps"])
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # Generate
    print(f"\nGenerating {args.num_samples} samples with {args.steps} steps...\n")
    generated = sample(
        model=model,
        noise_schedule=noise_schedule,
        seq_len=mc["seq_len"],
        num_samples=args.num_samples,
        num_steps=args.steps,
        device=args.device,
    )

    # Decode and print
    for i in range(args.num_samples):
        tokens = generated[i].cpu().tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"--- Sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
