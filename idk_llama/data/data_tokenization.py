"""
"Smart" dataset tokenization to be used in conjunction with `VeryCoolDataset` in `dataset.py`.
We tokenize each document and concatenate the token ids into one large np.memmap file, optionally adding BOS and EOS tokens.
We also store the start indices of each document in the concatenated file in a separate file.
That way, have O(1) access to each document and can retrieve samples with arbitrary sequence length without re-tokenization.
"""
import errno
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
from datasets import load_dataset
from simple_parsing import field, parse
from tqdm import tqdm
from transformers import AutoTokenizer


def get_cache_dir(source_dir: Path, tokenizer_path: Path) -> Path:
    return source_dir / "tokenized" / "_".join(tokenizer_path.parts)


def get_cache_paths(source_dir: Path, tokenizer_path: Path, split: str) -> tuple[Path, Path]:
    destination_path = get_cache_dir(source_dir, tokenizer_path)
    return (destination_path / f"{split}.bin", destination_path / f"{split}.bin.idx")


def check_cache(source_dir: Path, tokenizer_path: Path, splits: list[str]) -> bool:
    for split in splits:
        bin_path, idx_path = get_cache_paths(source_dir, tokenizer_path, split)
        if not bin_path.exists() or not idx_path.exists():
            return False
    return True


@dataclass
class TokenizationArgs:
    source_dir: Path
    "Source dir for the data. Expected to contain train_file and dev_file."

    train_file: str = "train.jsonl"
    dev_file: str = "dev.jsonl"
    destination_path: str | None = field(default=None)
    "If None, save to {source_dir}/tokenized/{tokenizer_path}"

    tokenizer_path: Path = field(default="checkpoints/meta-llama/Llama-2-7b-hf")
    num_proc: int = -1
    conserve_disk_space: bool = field(default=False)
    prepend_bos: bool = field(default=False)
    append_eos: bool = field(default=True)

    extra_val_clip_length: int | None = field(default=None)
    """If you want to compare PPL between models that use different tokenizers, 
    it is important that you do not normalize by the number of tokens but rather by some other constant that is the same for all models.
    One tricky detail is that if you use fixed length sequences, you need to make sure that all models actually have to predict the exact same text dataset.
    For models with a less specific tokenizer, this will mean more tokens.
    That's why with --extra_val_clip_length, we can create a clipped version of the val set that is the same for all models and can fit into the sequence length for all models.
    You should set this to some small number smaller than your sequence length, e.g. 1/4 of your sequence length."""

    just_clip_val: bool = field(default=False)
    """If True, only create the clipped val set and don't tokenize the train set."""


def tokenize_data(args: TokenizationArgs) -> None:
    if args.destination_path is None:
        destination_path = get_cache_dir(args.source_dir, args.tokenizer_path)
    destination_path.mkdir(parents=True, exist_ok=True)

    if check_cache(args.source_dir, args.tokenizer_path, ["train", "val"]):
        print(f"Warning: cache files in {get_cache_dir(args.source_dir, args.tokenizer_path)} already exist.")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # number of workers in .map() call
    # good number to use is ~order number of cpu cores // 2
    if args.num_proc == -1:
        args.num_proc = os.cpu_count() // 2

    data_files = {
        "train": str(args.source_dir / args.train_file),
        "val": str(args.source_dir / args.dev_file),
    }

    print("Loading raw dataset...")
    tmp_load_dataset_cache_dir = tempfile.mkdtemp(dir=str(destination_path)) if args.conserve_disk_space else None
    train_val_datasets = load_dataset(
        "json",
        data_files=data_files,
        name=f'{str(args.source_dir).replace("/", "_")}---{args.train_file}---{args.dev_file}---',
        num_proc=args.num_proc,
        cache_dir=tmp_load_dataset_cache_dir,
    )

    print(train_val_datasets, train_val_datasets["train"][:2])

    if args.conserve_disk_space:
        datasets.fingerprint.disable_caching()

    def process(example):
        ids = tokenizer(
            example["text"],
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        # Investigation: OpenLLama adds bos and eos to each sample
        # https://github.com/young-geng/EasyLM/blob/f926148081c9eddcc85df4a4d2277aa68a1efc47/EasyLM/data.py#L83
        # for bos, loss is masked. i.e. loss is not calculated when bos is the target. loss is calculated when eos is the target.
        # we take care of loss masking later in the dataset

        if args.prepend_bos:
            ids = [[tokenizer.bos_token_id] + sample for sample in ids]
        if args.append_eos:
            ids = [sample + [tokenizer.eos_token_id] for sample in ids]

        return {"ids": ids, "len": [len(sample) for sample in ids]}

    def clip_process(example):
        text = " ".join(example["text"].split(" ")[: args.extra_val_clip_length])
        ids = tokenizer(
            text,
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        # print(ids, len(ids), ids[0])
        return {"ids": ids, "len": len(ids)}

    if not args.just_clip_val:
        # tokenize the dataset
        tokenized = train_val_datasets.map(
            process,
            desc="tokenizing the splits",
            batched=True,
            batch_size=2**10,
            num_proc=args.num_proc,
            remove_columns=["text"],
        )

    clipped_tokenized_val = train_val_datasets["val"].map(
        clip_process,
        desc="tokenizing the val split",
        num_proc=args.num_proc,
        remove_columns=["text"],
    )
    # concatenate all the ids in each dataset into one large file we can use for training
    dset_splits = [] if args.just_clip_val else list(tokenized.items())
    if args.extra_val_clip_length is not None:
        dset_splits.append((f"val-clipped-{args.extra_val_clip_length}", clipped_tokenized_val))
    for split, dset in dset_splits:
        arr_len = np.sum(dset["len"], dtype=np.uint64)

        bin_path, idx_path = get_cache_paths(args.source_dir, args.tokenizer_path, split)
        DATA_DTYPE = np.uint16  # vocab size of e.g. 50256 is < 2**16 == 65536)
        if len(tokenizer) > 65536:
            print("Info: using uint32 for token ids since vocab size > 65536")
            DATA_DTYPE = np.uint32
        arr = np.memmap(str(bin_path), dtype=DATA_DTYPE, mode="w+", shape=(arr_len,))
        total_batches = 1024

        num_docs = len(dset)
        print("num docs", num_docs)
        # NOTE: uint64 can accomodate 2**64 = ~1.8e19 tokens, which is more than enough for our purposes
        OFFSET_DTYPE = np.uint64
        doc_offsets = np.memmap(
            str(idx_path),
            dtype=OFFSET_DTYPE,
            mode="w+",
            shape=(num_docs,),
        )

        token_idx = 0
        doc_idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {bin_path}"):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
            flattend_batch_tokens = np.concatenate(batch["ids"])

            # save offsets of doc start indices to use later on
            # first_len = batch["len"][0]
            # calculate relative start index of each sample to the batch start index
            offsets_to_prev = np.concatenate(([0], batch["len"][:-1]))
            offsets_to_start = np.cumsum(offsets_to_prev)
            # convert into absolute index
            offsets_to_start = offsets_to_start + token_idx

            num_docs_in_batch = len(batch["len"])

            # print(token_idx)
            # print(len(batch["len"]), len(offsets))
            # print(batch["len"][:100])
            # print(offsets)
            # print(sample_start_indices[:100])
            assert num_docs_in_batch == len(offsets_to_prev)
            assert num_docs_in_batch == len(offsets_to_start)
            assert offsets_to_start[0] == token_idx
            assert offsets_to_start[-1] == token_idx + len(flattend_batch_tokens) - batch["len"][-1]
            assert offsets_to_start[1] == token_idx + batch["len"][0]

            num_docs_in_batch = len(batch["len"])
            doc_offsets[doc_idx : doc_idx + num_docs_in_batch] = offsets_to_start
            doc_idx += num_docs_in_batch

            # Write into mmap
            arr[token_idx : token_idx + len(flattend_batch_tokens)] = flattend_batch_tokens
            token_idx += len(flattend_batch_tokens)

        doc_offsets.flush()
        arr.flush()

    metadata = {
        "train_data_file": get_cache_paths(args.source_dir, args.tokenizer_path, "train")[0].name,
        "train_index_file": get_cache_paths(args.source_dir, args.tokenizer_path, "train")[1].name,
        "dev_data_file": get_cache_paths(args.source_dir, args.tokenizer_path, "val")[0].name,
        "dev_index_file": get_cache_paths(args.source_dir, args.tokenizer_path, "val")[1].name,
        "tokenizer": args.tokenizer_path.as_posix(),
        "prepend_bos": args.prepend_bos,
        "bos_token_id": tokenizer.bos_token_id,
        "append_eos": args.append_eos,
        "eos_token_id": tokenizer.eos_token_id,
        "data_dtype": np.dtype(DATA_DTYPE).name,
        "doc_offset_dtype": np.dtype(OFFSET_DTYPE).name,
        "tokenizer_vocab": {t: i for t, i in sorted(tokenizer.get_vocab().items(), key=lambda x: x[1])},
    }

    if args.extra_val_clip_length is not None:
        val_clipped_bin_path, val_clipped_idx_path = get_cache_paths(
            args.source_dir, args.tokenizer_path, f"val-clipped-{args.extra_val_clip_length}"
        )
        metadata["val_clipped_data_file"] = val_clipped_bin_path.name
        metadata["val_clipped_index_file"] = val_clipped_idx_path.name

    # write metadata
    with open(destination_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    if args.conserve_disk_space:
        print("Cleaning dataset loading cache...")
        try:
            shutil.rmtree(tmp_load_dataset_cache_dir)
        except OSError as e:
            # Reraise unless ENOENT: No such file or directory
            # (ok if directory has already been deleted)
            if e.errno != errno.ENOENT:
                raise


if __name__ == "__main__":
    args = parse(TokenizationArgs)
    print(args)
    args.source_dir = Path(args.source_dir).resolve()
    args.destination_path = None if not args.destination_path else Path(args.destination_path).resolve()

    tokenize_data(args)
