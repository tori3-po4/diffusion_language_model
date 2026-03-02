"""FineWeb-Edu streaming data pipeline for MDLM pre-training."""

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class FineWebEduDataset(IterableDataset):
    """Streaming dataset that tokenizes FineWeb-Edu and yields fixed-length chunks.

    Reads documents via HF datasets streaming, tokenizes with GPT-2 tokenizer,
    concatenates all tokens into a buffer, and yields contiguous chunks of seq_len tokens.
    """

    def __init__(
        self,
        dataset_name: str = "HuggingFaceFW/fineweb-edu",
        dataset_config: str = "sample-10BT",
        seq_len: int = 256,
        shuffle_buffer: int = 10000,
        seed: int = 42,
    ):
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed

        self.tokenizer = AutoTokenizer.from_pretrained(
            "gpt2", model_max_length=int(1e9)
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed
        if worker_info is not None:
            seed += worker_info.id

        ds = load_dataset(
            self.dataset_name,
            self.dataset_config,
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=self.shuffle_buffer)

        buffer = []
        for example in ds:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len:
                yield {"input_ids": buffer[: self.seq_len]}
                buffer = buffer[self.seq_len :]


def get_dataloader(
    dataset_name: str = "HuggingFaceFW/fineweb-edu",
    dataset_config: str = "sample-10BT",
    seq_len: int = 256,
    batch_size: int = 128,
    shuffle_buffer: int = 10000,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    """Build a DataLoader for FineWeb-Edu streaming."""
    dataset = FineWebEduDataset(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seq_len=seq_len,
        shuffle_buffer=shuffle_buffer,
        seed=seed,
    )

    def collate_fn(batch):
        input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        return input_ids

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )
