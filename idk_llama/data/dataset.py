import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset as TorchDataset
from transformers import AutoTokenizer


def load_datasets(
    data_dir: Path,
    block_size: int,
    seed: int = 42,
    sanity_tokenizer: Path | None = None,
    use_clipped_val: bool = False,
):
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    print("Loading datasets...")
    # tokenizer vocab prints too much clutter
    print({k: v for k, v in metadata.items() if k != "tokenizer_vocab"})

    # do some sanity checks to make sure the user actually loaded the correct dataset
    data_tokenizer = AutoTokenizer.from_pretrained(metadata["tokenizer"])
    if sanity_tokenizer:
        # assert tokenizer == Path(metadata["tokenizer"])
        print("sanity tokenizer:", sanity_tokenizer)
        sanity_tokenizer = AutoTokenizer.from_pretrained(sanity_tokenizer)
        if not sanity_tokenizer.get_vocab() == data_tokenizer.get_vocab():
            print("Sanity tokenizer vocab does not match data tokenizer vocab. Using data tokenizer.")
            print(
                "token difference:",
                len(set(sanity_tokenizer.get_vocab()) ^ set(data_tokenizer.get_vocab())),
                set(sanity_tokenizer.get_vocab()) ^ set(data_tokenizer.get_vocab()),
            )
            print("additional tokens in data tokenizer:", set(data_tokenizer.get_vocab()) - set(sanity_tokenizer.get_vocab()))
            print("additional tokens in sanity tokenizer:", set(sanity_tokenizer.get_vocab()) - set(data_tokenizer.get_vocab()))

        assert sanity_tokenizer.pad_token_id == data_tokenizer.pad_token_id
        assert sanity_tokenizer.bos_token_id == data_tokenizer.bos_token_id
        assert sanity_tokenizer.eos_token_id == data_tokenizer.eos_token_id
        assert sanity_tokenizer.unk_token_id == data_tokenizer.unk_token_id

    metadata_tokenizer_vocab = metadata["tokenizer_vocab"]
    assert len(data_tokenizer.get_vocab()) == len(metadata_tokenizer_vocab)
    assert metadata_tokenizer_vocab == data_tokenizer.get_vocab()

    train_path = data_dir / metadata["train_data_file"]
    val_path = data_dir / metadata["dev_data_file"]
    train_index_path = data_dir / metadata["train_index_file"]
    val_idx_path = data_dir / metadata["dev_index_file"]

    if use_clipped_val:
        val_path = data_dir / metadata["val_clipped_data_file"]
        val_idx_path = data_dir / metadata["val_clipped_index_file"]

    assert train_path.exists()
    assert val_path.exists()
    assert train_index_path.exists()
    assert val_idx_path.exists()

    common_kwargs = dict(
        block_size=block_size,
        data_dtype=np.dtype(metadata["data_dtype"]),
        doc_offset_dtype=np.dtype(metadata["doc_offset_dtype"]),
        output_dtype=np.int64,  # expected by torch (int64 => long)
        bos_token=metadata["bos_token_id"],
        eos_token=metadata["eos_token_id"],
        mask_bos_loss=False,
        ensure_bos_token=False,
        pad_token=data_tokenizer.pad_token_id,
    )
    print("loading datasets internal")
    train_data = VeryCoolDataset(
        train_path,
        doc_offsets_file=train_index_path,
        shuffle=True,
        access="contiguous",
        **common_kwargs,
    )
    val_data = VeryCoolDataset(
        val_path,
        doc_offsets_file=val_idx_path,
        shuffle=False,
        full_samples=use_clipped_val,
        access="document-aware",
        # no_document_packing=use_clipped_val,
        **common_kwargs,
    )
    return train_data, val_data


def get_dataloaders(
    data_dir: Path,
    block_size: int,
    batch_size: int,
    workers: int,
    tokenizer_path: Path | None = None,
    val_batch_size: int | None = None,
    use_clipped_val: bool = False,
    resume_from_sample_idx: int | None = None,
):
    train_data, val_data = load_datasets(
        data_dir=data_dir,
        block_size=block_size,
        sanity_tokenizer=tokenizer_path,
        use_clipped_val=use_clipped_val,
    )
    if resume_from_sample_idx is not None:
        assert train_data.training_order is not None
        print(f"Resuming dataset from sample idx {resume_from_sample_idx}")
        train_data.training_order = train_data.training_order[resume_from_sample_idx:]
    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
        # persistent_workers=True, # Deactivate because we don't do more than one epoch anyways
        shuffle=False,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=val_batch_size or batch_size,
        num_workers=workers,
        pin_memory=True,
        # https://discuss.pytorch.org/t/what-are-the-dis-advantages-of-persistent-workers/102110/10
        # persistent_workers=True,
        shuffle=False,
        drop_last=False,
    )

    return train_dataloader, val_dataloader


class VeryCoolDataset(TorchDataset):
    """
    In `data_tokenization.py` we store the tokenized concatenated data (`data_file`) AND a file containing the indices in `data_file` where a new sample starts.
    We use this `index_file` to sample from the data so that the beginning of each sample aligns with the start of an actual sample in the data.

    This allows us to change the `block_size` with ZERO overhead at runtime (no expensive re-tokenization / chunking).
    However, we discard the remainder of each sample after the first `block_size` tokens.

    Expected format of each sample is a EOS token after each doc.

    *adapted heavily from lit-gpt pretrain/openwebtext.py*
    """

    def __init__(
        self,
        data_file: Path,
        doc_offsets_file: Path,
        block_size: int,
        access: Literal["contiguous", "document-aware"] = "document-aware",
        no_document_packing: bool = False,
        full_samples: bool = False,
        unk_token: int = 0,
        bos_token: int = 1,
        eos_token: int = 2,
        pad_token: int = -1,  # by default, chunked_cross_entropy ignores -1. llama2 does not have pad token in vocab
        ignore_index: int = -1,  # for cross entropy loss
        mask_bos_loss: bool = False,
        ensure_bos_token: bool = False,
        shuffle: bool = False,
        data_dtype: np.dtype = np.uint16,  # supports vocab size up to 65k
        doc_offset_dtype: np.dtype = np.uint64,  # supports up to 2**64 = a lot of tokens
        output_dtype: np.dtype = np.int64,  # for safety
    ):
        super().__init__()
        self.data_file = data_file
        self.block_size = block_size
        self.index_file = doc_offsets_file
        self.access = access
        self.no_document_packing = no_document_packing
        self.full_samples = full_samples
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.ignore_index = ignore_index
        self.mask_bos_loss = mask_bos_loss
        self.ensure_bos_token = ensure_bos_token

        self.data_dtype = data_dtype  # needs to address all token_ids in the vocab
        self.doc_offset_dtype = doc_offset_dtype  # needs to address all tokens in the dataset
        # output_dtype needs to represent data_dtype losslessly
        self.output_dtype = output_dtype  # needs to fit the entire vocab range. torch.from_numpy wants intXX, not uintXX

        self.data = np.memmap(self.data_file, dtype=self.data_dtype, mode="r")
        self.doc_offsets = np.memmap(self.index_file, dtype=self.doc_offset_dtype, mode="r")

        self.num_samples = self.doc_offsets.size
        if self.access == "contiguous":
            self.num_samples = self.data.size // self.block_size

        self.training_order = None
        if shuffle:
            self.training_order = self.get_reproducible_shuffled_training_order()

        if self.no_document_packing:
            assert self.access == "document-aware"
            assert self.full_samples is False

        if self.full_samples:
            assert self.access == "document-aware"
            assert self.no_document_packing is False

    def get_reproducible_shuffled_training_order(self, seed: int = 42):
        """
        Write a .npy file containing the shuffled indices for reproducible and resumable training.
        """
        assert self.num_samples is not None

        cache_path = self.data_file.with_suffix(f".shuffled_idx_w_seed_{seed}_n_{self.num_samples}.npy")

        if not cache_path.exists():
            # needs to address number of *samples (documents)* in the dataset, which is < 2**32 ~ 4.3B
            sample_idx_dtype = np.uint32
            training_order = np.arange(start=0, stop=self.num_samples, step=1, dtype=sample_idx_dtype)
            rng = np.random.default_rng(seed=seed)
            rng.shuffle(training_order)
            np.save(cache_path, training_order)

        print(f"Loading cached shuffled indices from {cache_path}")
        training_order = np.load(cache_path, mmap_mode="r")
        return training_order

    def __len__(self) -> int:
        return self.num_samples

    def _read_data(self, start_idx: int, end_idx: int) -> torch.Tensor:
        return torch.from_numpy((self.data[start_idx:end_idx]).astype(self.output_dtype))

    def _mask(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.mask_bos_loss:
            y[y == self.bos_token] = self.ignore_index
        y[x == self.pad_token] = self.ignore_index  # always ignore pad tokens - never learn loss *on* pad tokens
        y[y == self.pad_token] = self.ignore_index  # always ignore pad tokens - never learn to pred pad tokens
        y[x == self.eos_token] = self.ignore_index  # never learn loss *on* EOS tokens - but learn to pred EOS
        return x, y

    def _maybe_ensure_bos_token(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ensure_bos_token:
            if x[0] != self.bos_token:
                y = x
                x = torch.cat([torch.tensor([self.bos_token], dtype=x.dtype), x[:-1]])
        return x, y

    def _contiguous_access(self, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Just extract a contiguous block of tokens from the data file, not respecting any document boundaries.
        EOS tokens should have already been inserted after each document.
        """
        data_idx = sample_idx * self.block_size
        x = self._read_data(data_idx, data_idx + self.block_size)
        y = self._read_data(data_idx + 1, data_idx + self.block_size + 1)
        x, y = self._maybe_ensure_bos_token(x, y)
        return self._mask(x, y)

    def _full_samples_access(self, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        NOTE: this does not respect the block_size, but returns the full sample.
        """
        doc_data_idx = self.doc_offsets[sample_idx].item()
        next_doc_data_idx = self.doc_offsets[sample_idx + 1].item()
        x = self._read_data(doc_data_idx, next_doc_data_idx - 1)
        y = self._read_data(doc_data_idx + 1, next_doc_data_idx)
        if self.ensure_bos_token:
            if x[0] != self.bos_token:
                y = torch.cat([x[0:1], y])  # Shift y one to the left (i.e. one token earlier)
                x = torch.cat([torch.tensor([self.bos_token], dtype=x.dtype), x])
        return self._mask(x, y)  # NOTE: should we mask here?

    def _doc_start_contiguous_access(self, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract a contiguous block of tokens from the data file, but always starting at the beginning of a document.
        """
        doc_data_idx = self.doc_offsets[sample_idx].item()
        x = self._read_data(doc_data_idx, doc_data_idx + self.block_size)
        y = self._read_data(doc_data_idx + 1, doc_data_idx + 1 + self.block_size)
        x, y = self._maybe_ensure_bos_token(x, y)
        return self._mask(x, y)

    def _doc_start_single_doc_access(self, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract up to `block_size` tokens from the data file, always starting at the beginning of a document.
        When a document is shorter than `block_size`, the remainder is padded with the pad token.
        """
        doc_data_idx = self.doc_offsets[sample_idx].item()
        x = self._read_data(doc_data_idx, doc_data_idx + self.block_size)
        y = self._read_data(doc_data_idx + 1, doc_data_idx + 1 + self.block_size)

        next_doc_data_idx = self.doc_offsets[sample_idx + 1].item()
        doc_len = next_doc_data_idx - doc_data_idx
        if doc_len < self.block_size:
            # padding - overwrite the remainder that was part of the next sample
            # put at least 0 because some models don't have pad token, in which case we use -1 for targets
            # inputs get ignored anyway
            x[doc_len:] = max(self.pad_token, 0)
            # for y, it's shifted one to the right, so next sample starts one token earlier
            y[doc_len - 1 :] = self.ignore_index
        x, y = self._maybe_ensure_bos_token(x, y)
        return self._mask(x, y)

    def __getitem__(self, sample_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training_order is not None:
            sample_idx = self.training_order[sample_idx].item()

        ######### Contiguous access #########
        if self.access == "contiguous":
            return self._contiguous_access(sample_idx)
        elif self.access == "document-aware":
            if self.full_samples:
                return self._full_samples_access(sample_idx)

            if self.no_document_packing:
                return self._doc_start_single_doc_access(sample_idx)
            else:
                return self._doc_start_contiguous_access(sample_idx)
        else:
            raise ValueError(f"Unknown access type: {self.access}")
