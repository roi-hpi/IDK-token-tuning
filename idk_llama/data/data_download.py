"""
This script allows you to download and preprocess datasets for language modeling, specifically mc4 and cc100. You can customize it to your own needs.

Example command to download cc100 for German:
python preprocess_data.py --lg de --dataset cc100 --out_dir ./data/cc100/ --processes 8


Example command to download cc100 for German using streaming mode for HF datasets (faster, requires less RAM) and cleaning up caches:
python preprocess_data.py --lg de --dataset cc100 --out_dir ./data/cc100/ --processes 8 --stream --stream_shuffle_buffer_size 1_000_000 --conserve_disk_space

Inspiration from lit-gpt and gpt-neox.
"""

import errno
import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal

import datasets
import jsonlines
from datasets import load_dataset
from print_on_steroids import graceful_exceptions, logger
from simple_parsing import field, parse
from tqdm import tqdm


@dataclass
class Args:
    language: str = field(alias="--lg")
    "Language identifier"

    out_dir: str = field(alias="-o")

    dataset: Literal["mc4", "cc100", "oscar2023", "pile"] = field(default="oscar2023")
    "HF dataset. Pile currently uses a mirror with copyrighted material removed."

    max_train_size: int = field(default=50_000_000)
    "Maximum number of train documents to write to disk. Use to limit very large datasets that you will not exhaust during training anyway. Use -1 to disable."

    dev_size: int = field(default=25_000)
    "If 0, do not construct dev set."

    test_size: int = field(default=25_000)
    "If 0, do not contruct test set."

    processes: int = field(default=4)
    "Number of processes for parallel tokenization."

    split: str = field(default="train")
    "Select percentage of dataset like so: --split=train[:50%]"

    conserve_disk_space: bool = field(default=False, alias="--disk_space")
    "Disable all HF caching and cleanup download caches to conserve disk space."

    stream: bool = field(default=True)
    "Couple with max_train_size to avoid having to load the entire dataset. Use streaming mode to load dataset and process dataset."

    stream_shuffle_buffer_size: int = field(default=100_000)
    """Buffer size for shuffling datasets before splitting in streaming mode.
    The entire buffer will be downloaded before shuffling.
    You also need to have enough RAM if you set this to a large value If -1, set to max_train_size."""

    pre_discard_factor: float = field(default=None)
    """Percentage of the dataset to discard before any processing.
    Useful for speeding up processing of huge datasets that are not fully needed.
    Not needed if you use --stream."""

    format: Literal["txt", "jsonl"] = field(default="jsonl")
    "Format to write to disk. Prefer jsonl over txt for better handling of newlines in documents and because it can be laoded much faster by HF datasets."


@graceful_exceptions()
def main(args: Args):
    logger.info(args)
    if args.max_train_size == -1:
        args.max_train_size = None
    if args.conserve_disk_space:
        # Disable caching because we write the end result to disk anyways. Intermediary caches just clutter the disk!
        logger.info("Disabling caching to conserve disk space.")
        datasets.fingerprint.disable_caching()

    os.makedirs(args.out_dir, exist_ok=True)
    logger.info("Downloading dataset. This can take some time, so sit back and relax...")

    tmp_cache_dir = None
    if args.conserve_disk_space:
        tmp_cache_dir = os.path.join(args.out_dir, args.language, "tmp_download_cache")
        os.makedirs(tmp_cache_dir, exist_ok=True)

    ##### Load dataset #####
    if args.dataset == "mc4":
        dataset = load_dataset(
            "mc4",
            args.language,
            split=args.split,
            cache_dir=tmp_cache_dir,
            streaming=args.stream,
            num_proc=None if args.stream else args.processes,
        )
    elif args.dataset == "cc100":
        if args.stream:
            logger.warning("Streaming mode for cc100 might lose some sample documents.")
            # Streaming mode is not trivial for cc100, since we need to group samples into documents.
            # To lose no samples, we need to set batch_size=len(dataset) but this is not possible for IteratableDataset.
            # We can accept loosing some samples by setting batch_size to a large number.
        dataset = load_dataset(
            "cc100",
            lang=args.language,
            split=args.split,
            cache_dir=tmp_cache_dir,
            streaming=args.stream,
            num_proc=None if args.stream else args.processes,
        )
    elif args.dataset == "oscar2023":
        dataset = load_dataset(
            "oscar-corpus/OSCAR-2301",
            language=args.language,
            split=args.split,
            cache_dir=tmp_cache_dir,
            streaming=args.stream,
            token=True,
            num_proc=None if args.stream else args.processes,
        )
        print(dataset)

        # if args.dataset == "oscar2023":
        # For oscar2023, we need to rename the columns to match the other datasets
        # dataset = dataset.rename_column("content", "text")

        # Filter out all samples with content warning in OSCAR
        dataset = dataset.filter(
            lambda x: x["meta"]["quality_warnings"] is None
            or (
                "noisy" not in x["meta"]["quality_warnings"]
                and "header" not in x["meta"]["quality_warnings"]
                and "footer" not in x["meta"]["quality_warnings"]
                and "short_sentences" not in x["meta"]["quality_warnings"]
                and "tiny" not in x["meta"]["quality_warnings"]
                and "adult" not in x["meta"]["quality_warnings"]
            ),
        )
        print(dataset)
    elif args.dataset == "pile":
        dataset = load_dataset(
            "monology/pile-uncopyrighted",
            split=args.split,
            cache_dir=tmp_cache_dir,
            streaming=args.stream,
            num_proc=None if args.stream else args.processes,
        )
        print(dataset)

    ##### For CC100: Group individual lines into documents #####
    if args.dataset == "cc100":

        def document_grouping_f(examples: Dict[str, list[str]]):
            documents = []
            current_doc = ""
            for example in examples["text"]:
                if example == "\n":
                    documents.append(current_doc)
                    current_doc = ""
                else:
                    current_doc += example
            return {"docs": documents}

        batch_size = 16_000 if args.stream else len(dataset)
        map_args = dict(batched=True, batch_size=batch_size, remove_columns=["text", "id"])
        if not args.stream:
            map_args["num_proc"] = args.processes  # stream does not support multiprocessing
        dataset = dataset.map(document_grouping_f, **map_args)
        dataset = dataset.rename_column("docs", "text")

    ##### If not streaming, optionally pre-discard parts of the dataset (for faster processing) #####
    if not args.stream:
        dataset_len = len(dataset)
        logger.info("Dataset len:", dataset_len)

        if args.pre_discard_factor:
            assert args.max_train_size is not None
            if dataset_len > args.pre_discard_factor * args.max_train_size:
                dataset = dataset.shuffle(seed=42).select(range(int(args.pre_discard_factor * args.max_train_size)))

    logger.info(dataset)

    #### Define processing functions ####

    def basic_f(examples: Dict[str, list[str]]):
        return {"processed_text": [example.strip() + "\n" for example in examples["text"] if example != "\n"]}

    f = basic_f

    ##### Process dataset #####
    if args.stream:
        # dataset.columns_names is not available for streaming datasets #TODO: this is fixed in the current master version of datasets
        cols = ["text"]
        if args.dataset == "mc4":
            cols.extend(["timestamp", "url"])
        if args.dataset == "pile":
            cols.append("meta")
        dataset = dataset.map(f, batch_size=16_000, batched=True, remove_columns=cols)
    else:
        logger.info("Starting mapping & chunking")
        dataset = dataset.map(
            f,
            num_proc=args.processes,
            batch_size=16_000,
            batched=True,
            remove_columns=dataset.column_names,
        )
        logger.success("Processing finished!")

    ##### Split into train/dev/test #####
    logger.info("Shuffling and splitting into sets...")
    # Preferred for big datasets.
    if args.stream:
        # Careful here, this does not truly shuffle ALL the data by default, only samples within a buffer
        # You might have to adjust the buffer_size here depending on memory limits of your machine
        # Then take care of true shuffling in the Dataloader
        args.stream_shuffle_buffer_size = None if args.stream_shuffle_buffer_size == -1 else args.stream_shuffle_buffer_size
        logger.debug(f"Shuffling with buffer size {args.stream_shuffle_buffer_size}")
        dataset = dataset.shuffle(seed=42, buffer_size=args.stream_shuffle_buffer_size)

        if args.dev_size:
            logger.debug(f"Taking {args.dev_size} dev samples")
            dev_paragraphs = dataset.take(args.dev_size)
            dataset = dataset.skip(args.dev_size)

        if args.test_size:
            logger.debug(f"Taking {args.test_size} test samples")
            test_paragraphs = dataset.take(args.test_size)
            dataset = dataset.skip(args.test_size)

        logger.debug(f"Taking {args.max_train_size} train samples")
        train_paragraphs = dataset.take(args.max_train_size)

        logger.info(f"Example train split data: {list(train_paragraphs.take(4))}")
    else:
        total_len = len(dataset)
        logger.info(f"Dataset len after processing: {total_len}")

        dataset = dataset.shuffle(seed=42)

        dev_test_size = args.dev_size + (args.test_size or 0)
        train_end_idx = total_len - dev_test_size
        train_paragraphs = dataset.select(range(train_end_idx))
        if args.max_train_size and len(train_paragraphs) > args.max_train_size:
            train_paragraphs = train_paragraphs.select(range(args.max_train_size))

        dev_paragraphs = dataset.select(range(train_end_idx, train_end_idx + args.dev_size))
        if args.test_size:
            test_paragraphs = dataset.select(range(train_end_idx + args.dev_size, total_len))
        logger.info(f"Example train split data: {train_paragraphs[:4]}")
        logger.info(f"len: {len(train_paragraphs)}")

    if args.conserve_disk_space:
        logger.info("Cleaning download cache")
        try:
            shutil.rmtree(tmp_cache_dir)
        except OSError as e:
            # Reraise unless ENOENT: No such file or directory
            # (ok if directory has already been deleted)
            if e.errno != errno.ENOENT:
                raise

    ##### Write to disk #####
    logger.info("Writing data...")
    output_dir = Path(args.out_dir) / args.language
    if args.dataset == "pile":
        output_dir = Path(args.out_dir)  # Pile is monolingual. We can think about subsets later.
    os.makedirs(str(output_dir), exist_ok=True)
    PERFORMANT_BUFFER_SIZE_BYTES = 1024 * 1024 * 100  # 100 MB

    # Preferred.
    if args.format == "jsonl":
        train_fp = io.open(str(output_dir / "train.jsonl"), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES)
        with jsonlines.Writer(train_fp, compact=True) as writer:
            writer.write_all(({"text": t["processed_text"]} for t in tqdm(train_paragraphs, desc="Writing train data...")))
        train_fp.close()

        if args.dev_size:
            dev_fp = io.open(str(output_dir / "dev.jsonl"), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES)
            with jsonlines.Writer(dev_fp, compact=True) as writer:
                writer.write_all(({"text": t["processed_text"]} for t in tqdm(dev_paragraphs, desc="Writing dev data...")))
            dev_fp.close()

        if args.test_size:
            test_fp = io.open(str(output_dir / "test.jsonl"), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES)
            with jsonlines.Writer(test_fp, compact=True) as writer:
                writer.write_all(({"text": t["processed_text"]} for t in tqdm(test_paragraphs, desc="Writing test data...")))
            test_fp.close()

    # Legacy, not preferred anymore. Doesn't handle newlines in documents well.
    elif args.format == "txt":
        with open(str(output_dir / "train.txt"), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES) as file:
            file.writelines((t["processed_text"] for t in tqdm(train_paragraphs, desc="Writing train data...")))

        with open(str(output_dir / "dev.txt"), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES) as file:
            file.writelines((t["processed_text"] for t in tqdm(dev_paragraphs, desc="Writing dev data...")))
        if args.test_size:
            with open(str(output_dir / "test.txt"), "wt", buffering=PERFORMANT_BUFFER_SIZE_BYTES) as file:
                file.writelines((t["processed_text"] for t in tqdm(test_paragraphs, desc="Writing test data...")))

    logger.success("Done! Enjoy your data :)")
    logger.print(output_dir / "train.jsonl")


if __name__ == "__main__":
    args = parse(Args)
    main(args)
