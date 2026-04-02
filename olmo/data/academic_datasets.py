import json
import logging
import re
from collections import defaultdict
from os.path import exists
from os.path import join

import datasets
import numpy as np
from pathlib import Path

from olmo.data.dataset import DATA_HOME, DatasetBase, Dataset, HfDataset
from olmo.hf_datasets.a_okvqa import AOkVqaBuilder
from olmo.hf_datasets.ai2d import Ai2dDatasetBuilder
from olmo.hf_datasets.android_control import AndroidControlBuilder
from olmo.hf_datasets.clock_bench import ClockBenchBuilder
from olmo.hf_datasets.count_qa import CountQaBuilder
from olmo.hf_datasets.dv_qa import DvQaBuilder
from olmo.hf_datasets.figure_qa import FigureQaBuilder
from olmo.hf_datasets.plot_qa import PlotQaBuilder
from olmo.hf_datasets.tabmwp import TabMwpBuilder
from olmo.hf_datasets.tally_qa import TallyQaBuilder
from olmo.hf_datasets.vqa_v2 import VQAv2BuilderMultiQA
from datasets import Dataset, Image
import logging
from datasets.utils.file_utils import DownloadConfig
from typing import Optional
import os 


import os, json, random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
import os, io, pandas as pd
from glob import glob



if DATA_HOME is not None:
    DOWNLOADS = join(DATA_HOME, "downloads")
    INFOQA_SOURCE = join(DATA_HOME, "info_qa")
    ST_QA_SRC = join(DATA_HOME, "scene-text")
else:
    DOWNLOADS = None
    INFOQA_SOURCE = None
    ST_QA_SRC = None


# class ChartQa(HfDataset):
#     """
#     ChartQA dataset from HuggingFace M4 project.
#     This class loads the ChartQA dataset from HuggingFace (https://huggingface.co/datasets/HuggingFaceM4/ChartQA).

#     Args:
#         split (str): Dataset split to load. One of "train", "validation", or "test".
#         parts (str, optional): Which subset of examples to include. One of:
#             - "human": Only human-authored examples
#             - "augmented": Only automatically generated examples
#             - "both": Both human and augmented examples (default)
#         weighted (bool, optional): Whether to apply weighting to balance human/augmented examples. Only valid when parts="both".
#             Defaults to False.
#     """
#     PATH = "HuggingFaceM4/ChartQA"

#     def __init__(self, split: str, parts="both", weighted=False, keep_in_memory=False):
#         assert split in ["train", "validation", "test"]
#         assert parts in ["human", "augmented", "both"]

#         if split == "validation":
#             split = "val"
#         self.updated_split = split
#         self.weighted = weighted
#         self.parts = parts
#         super().__init__(split, keep_in_memory=keep_in_memory)
#         if self.parts != "both":
#             # Filter out either human or aug datasets
#             flags = [int(self.parts == "human")]
#             self.dataset = self.dataset.filter(
#                 lambda x: x in flags,
#                 input_columns=["human_or_machine"]
#             )

#     def get(self, item, rng):
#         ex = self.dataset[item]
#         ex = dict(
#             image=ex["image"],
#             question=ex["query"],
#             answers=ex["label"],
#             style="chart_qa",
#             metadata=dict(
#                 is_human=ex['human_or_machine'],
#             )
#         )
#         if self.weighted:
#             is_human = ex["metadata"]["is_human"]
#             # Weight to balanced human/augmented sets
#             if is_human:
#                 w = 2*20901/(20901+7398)
#             else:
#                 w = 2*7398/(20901+7398)
#             ex["weight"] = w
#         return ex



# class ChartQa(HfDataset):
#     """
#     ChartQA dataset from HuggingFace M4 project.
#     This class loads the ChartQA dataset from HuggingFace (https://huggingface.co/datasets/HuggingFaceM4/ChartQA).

#     Args:
#         split (str): "train", "validation", or "test".
#         parts (str, optional): "human" | "augmented" | "both" (default "both").
#         weighted (bool, optional): If True (only valid when parts="both"), add class-balancing weight.
#         eval_train_when_val (bool, optional): If True and split=="validation", load/evaluate on train instead.
#         debug_train_n (int | None, optional): If set and the effective split is 'train', keep only first N samples.
#         keep_in_memory (bool, optional): Passed to datasets.load_dataset.
#     """
#     PATH = "HuggingFaceM4/ChartQA"

#     def __init__(
#         self,
#         split: str,
#         parts: str = "both",
#         weighted: bool = False,
#         eval_train_when_val: bool = False,
#         debug_train_n: int | None = 400,
#         keep_in_memory: bool = False,
#     ):
#         assert split in ["train", "validation", "test"], f"Invalid split: {split}"
#         assert parts in ["human", "augmented", "both"], f"Invalid parts: {parts}"

#         self.weighted = weighted
#         self.parts = parts
#         self.eval_train_when_val = eval_train_when_val
#         self.debug_train_n = debug_train_n

#         # If requesting train split, merge all available splits
#         if split == "train":
#             datasets_to_merge = []
            
#             # Try to load all splits (ChartQA uses "val" instead of "validation")
#             for s in ["train", "val", "test"]:
#                 try:
#                     ds = datasets.load_dataset(
#                         self.PATH,
#                         split=s,
#                         keep_in_memory=keep_in_memory
#                     )
#                     if len(ds) > 0:
#                         datasets_to_merge.append(ds)
#                         print(f"[ChartQa] Loaded {s} split with {len(ds)} examples")
#                 except Exception as e:
#                     print(f"[ChartQa] Could not load {s} split: {e}")
            
#             # Merge all loaded splits
#             if len(datasets_to_merge) > 1:
#                 from datasets import concatenate_datasets
#                 merged_dataset = concatenate_datasets(datasets_to_merge)
#                 print(f"[ChartQa] Merged {len(datasets_to_merge)} splits into train: total {len(merged_dataset)} examples")
                
#                 # Initialize parent with train split first
#                 self.updated_split = "train"
#                 super().__init__("train", keep_in_memory=keep_in_memory)
#                 # Replace with merged dataset
#                 self.dataset = merged_dataset
#             elif len(datasets_to_merge) == 1:
#                 self.updated_split = "train"
#                 super().__init__("train", keep_in_memory=keep_in_memory)
#             else:
#                 raise ValueError("[ChartQa] No data found in any split")
                
#         else:
#             # For validation/test, return empty dataset
#             hf_split = "val" if split == "validation" else split
#             self.updated_split = hf_split
            
#             # Load train first then make it empty
#             super().__init__("train", keep_in_memory=keep_in_memory)
#             self.dataset = self.dataset.select([])  # Empty dataset
            
#             # Alternative: Keep original split (uncomment if needed)
#             # effective_split = "train" if (split == "validation" and eval_train_when_val) else hf_split
#             # self.updated_split = hf_split
#             # super().__init__(effective_split, keep_in_memory=keep_in_memory)

#         # Optionally restrict to first N when we're on train (only applies if split was "train")
#         if split == "train" and self.debug_train_n is not None:
#             original_len = len(self.dataset)
#             self.dataset = self.dataset.select(range(min(self.debug_train_n, len(self.dataset))))
#             print(f"[ChartQa] Debug mode: reduced from {original_len} to {len(self.dataset)} examples")

#         # Optional filtering by parts (human vs augmented)
#         if self.parts != "both":
#             want_flag = 1 if self.parts == "human" else 0
#             original_len = len(self.dataset)
#             self.dataset = self.dataset.filter(
#                 lambda flag: flag == want_flag,
#                 input_columns=["human_or_machine"]
#             )
#             print(f"[ChartQa] Filtered to {self.parts}: {original_len} -> {len(self.dataset)} examples")

#     def get(self, item, rng):
#         ex = self.dataset[item]
#         ex = dict(
#             image=ex["image"],
#             question=ex["query"],
#             answers=ex["label"],
#             style="chart_qa",
#             metadata=dict(
#                 is_human=ex["human_or_machine"],
#             ),
#         )
#         if self.weighted and self.parts == "both":
#             # Balance human vs augmented with fixed counts (adjust if your corpus counts differ)
#             is_human = bool(ex["metadata"]["is_human"])
#             # These constants come from your snippet; keep them here for reproducibility
#             human_n, machine_n = 20901, 7398
#             if is_human:
#                 w = 2 * human_n / (human_n + machine_n)
#             else:
#                 w = 2 * machine_n / (human_n + machine_n)
#             ex["weight"] = w
#         return ex


import os, io, base64
from glob import glob
from typing import Optional
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ChartQa(Dataset):
    """
    Parquet-backed ChartQA dataset.


    Columns:
      - image: dict{'bytes'} OR bytes/bytearray/memoryview OR base64 str OR path
      - query: str
      - label: str or list[str]
      - human_or_machine: 1 for human, 0 for augmented (optional)
    """

    # CHANGE THIS to your actual directory if needed, or set env CHARTQA_DIR
    DEFAULT_DIR = os.environ.get(
        "CHARTQA_DIR",
        "/Document/molmo_code/data/torch_datasets/ChartQA",
    )

    def __init__(
        self,
        split: str,
        parts: str = "both",              # "human" | "augmented" | "both"
        weighted: bool = False,
        debug_train_n: Optional[int] = None,
        data_dir: Optional[str] = None,   # optional; if omitted, uses DEFAULT_DIR
    ):
        # Normalize split
        split_norm = split.lower()
        if split_norm == "validation":
            split_norm = "val"
        assert split_norm in ["train", "val", "test"], f"Invalid split: {split}"

        # Determine data_dir robustly
        # If caller passed "validation" (or any split string) as data_dir by mistake,
        # ignore it and use DEFAULT_DIR.
        if data_dir is None or data_dir.lower() in ("train", "val", "validation", "test"):
            data_dir = self.DEFAULT_DIR

        if not os.path.isdir(data_dir):
            raise FileNotFoundError(
                f"[ChartQa] data_dir not found: {data_dir}. "
                f"Set CHARTQA_DIR or pass data_dir explicitly."
            )

        self.data_dir = data_dir
        self.split = split_norm
        self.parts = parts
        self.weighted = weighted
        self.debug_train_n = debug_train_n

        self._load_parquet_shards()
        self._maybe_filter_parts()
        self._maybe_debug_subsample()

    # ---------- loading ----------
    def _load_parquet_shards(self):
        pat = os.path.join(self.data_dir, f"{self.split}-*.parquet")
        files = sorted(glob(pat))

        # For safety: if split is "val" but files are named "validation-*.parquet"
        if not files and self.split == "val":
            alt = sorted(glob(os.path.join(self.data_dir, "validation-*.parquet")))
            if alt:
                files = alt

        if not files:
            existing = sorted(glob(os.path.join(self.data_dir, "*.parquet")))
            msg = (
                f"No parquet files found for split '{self.split}' in {self.data_dir}\n"
                f"Tried pattern: {pat}\n"
                f"Found {len(existing)} parquet file(s) in dir:\n  " + "\n  ".join(existing[:20])
            )
            raise FileNotFoundError(msg)

        dfs = [pd.read_parquet(p) for p in files]
        self.df = pd.concat(dfs, ignore_index=True)
        print(f"[ChartQa] Loaded {len(self.df)} rows from {len(files)} file(s) for split='{self.split}' in {self.data_dir}")

    def _maybe_filter_parts(self):
        if self.parts == "both":
            return
        want = 1 if self.parts == "human" else 0
        if "human_or_machine" in self.df.columns:
            before = len(self.df)
            self.df = self.df[self.df["human_or_machine"] == want].reset_index(drop=True)
            print(f"[ChartQa] Filter parts='{self.parts}': {before} -> {len(self.df)}")
        else:
            print("[ChartQa] 'human_or_machine' column not present; skipping parts filter")

    def _maybe_debug_subsample(self):
        if self.split == "train" and self.debug_train_n is not None:
            before = len(self.df)
            self.df = self.df.iloc[: min(self.debug_train_n, len(self.df))].reset_index(drop=True)
            print(f"[ChartQa] Debug subset: {before} -> {len(self.df)}")

    # ---------- torch dataset ----------
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.get(idx)

    # ---------- helpers ----------
    def _decode_image(self, payload):
        import io
        if isinstance(payload, dict) and "bytes" in payload:
            raw = payload["bytes"]
            if isinstance(raw, memoryview): raw = raw.tobytes()
            elif isinstance(raw, bytearray): raw = bytes(raw)
            return Image.open(io.BytesIO(raw)).convert("RGB")

        if isinstance(payload, (bytes, bytearray, memoryview)):
            raw = bytes(payload)
            return Image.open(io.BytesIO(raw)).convert("RGB")

        if isinstance(payload, str):
            # path vs base64
            if os.path.exists(payload):
                with open(payload, "rb") as f:
                    raw = f.read()
                return Image.open(io.BytesIO(raw)).convert("RGB")
            # try base64
            try:
                s = payload
                if s.startswith("data:"):
                    s = s.split(",", 1)[-1]
                raw = base64.b64decode(s, validate=False)
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as _:
                pass

        raise TypeError(f"Unsupported image payload type: {type(payload)}")

    def _answers_list(self, v):
        if v is None:
            return [""]
        if isinstance(v, list):
            return [str(x) for x in v]
        return [str(v)]

    # ---------- public API ----------
    def get(self, item, rng=None):
        row = self.df.iloc[item]
        image = self._decode_image(row["image"])
        question = row.get("query", "")
        answers = self._answers_list(row.get("label", ""))

        ex = dict(
            image=image,
            question=question,
            answers=answers,
            style="chart_qa",
            metadata=dict(
                is_human=int(row["human_or_machine"]) if "human_or_machine" in row else None,
            ),
        )

        if self.weighted and self.parts == "both":
            # same constants you had; adjust if your corpus counts differ
            human_n, machine_n = 20901, 7398
            is_human = bool(ex["metadata"]["is_human"]) if ex["metadata"]["is_human"] is not None else True
            if is_human:
                w = 2 * human_n / (human_n + machine_n)
            else:
                w = 2 * machine_n / (human_n + machine_n)
            ex["weight"] = w

        return ex
    
    

class Vqa2(Dataset):
    @classmethod
    def download(cls, n_procs=1):
        VQAv2BuilderMultiQA(DOWNLOADS).download_and_prepare()

    def __init__(self, split, multi_question=False):
        assert split in ["train", "validation", "test"]
        self.multi_question = multi_question
        self.dataset = VQAv2BuilderMultiQA(DOWNLOADS).as_dataset(split=split)
        if not self.multi_question:
            flattened_data = []
            for item in self.dataset:
                for q in item["messages"]:
                    flattened_data.append(dict(
                        style=q['style'],
                        question=q["question"],
                        answers=q["answers"],
                        image=item["image"],
                        image_id=item["image_id"],
                        question_id=q["question_id"],
                    ))
            self.dataset = flattened_data

    def __len__(self):
        return len(self.dataset)

    def _drop_zip_dir_component(self, p):
        """
        Remove '.zip' suffix from any directory component in the path.
        Fast, no regex. Works for strings or Path-like objects.
        """
        if not isinstance(p, (str, Path)):
            return p
        s = str(p)
        parts = s.split('/')  # posix-style; your dataset paths use '/'
        for i, comp in enumerate(parts):
            if comp.endswith('.zip'):
                parts[i] = comp[:-4]  # strip ".zip"
        return '/'.join(parts)

    def get(self, item, rng):
        ex = self.dataset[item]

        # Normalize the image path (e.g., remove ".../train2014.zip/" -> ".../train2014/")
        img = ex.get("image", None)
        if isinstance(img, (str, Path)):
            img = self._drop_zip_dir_component(img)

        if self.multi_question:
            return {
                "metadata": {"image_id": ex["image_id"]},
                "image": img,
                "message_list": ex["messages"],
            }
        else:
            return {
                "style": "vqa2",
                "answers": ex["answers"],
                "metadata": {"image_id": ex["image_id"], "example_id": ex["question_id"]},
                "image": img,
                "question": ex["question"],
            }

            
            
class AOkVqa(Dataset):
    @classmethod
    def download(cls, n_procs=1):
        AOkVqaBuilder(DOWNLOADS).download_and_prepare()

    def __init__(self, split, direct_answer=False):
        self.split = split
        self.direct_answer = direct_answer
        self.style = "a_okvqa_" + ("da" if direct_answer else "mc")
        
        # Load and merge splits if requesting train
        if split == "train":
            # Get all available splits from the builder
            builder = AOkVqaBuilder(DOWNLOADS)
            available_splits = list(builder.info.splits.keys())
            
            # Load all available splits
            datasets_to_merge = []
            for s in available_splits:
                try:
                    ds = builder.as_dataset(split=s)
                    if len(ds) > 0:
                        datasets_to_merge.append(ds)
                        print(f"Loaded {s} split with {len(ds)} examples")
                except Exception as e:
                    print(f"Could not load split {s}: {e}")
            
            # Concatenate all splits
            if len(datasets_to_merge) > 1:
                from datasets import concatenate_datasets
                self.dataset = concatenate_datasets(datasets_to_merge)
                print(f"Merged {len(datasets_to_merge)} splits into train: total {len(self.dataset)} examples")
            elif len(datasets_to_merge) == 1:
                self.dataset = datasets_to_merge[0]
            else:
                raise ValueError("No data found in any split")
        else:
            # For validation/test, return empty dataset or keep original behavior
            # Option 1: Return empty dataset
            self.dataset = AOkVqaBuilder(DOWNLOADS).as_dataset(split="train").select([])
            
            # Option 2: Keep original split (uncomment if you want to keep val/test separate)
            # self.dataset = AOkVqaBuilder(DOWNLOADS).as_dataset(split=split)
        
        self.loaded_data = self.load()

    def _drop_zip_dir_component(self, p):
        """
        Remove '.zip' suffix from any directory component in the path.
        Fast, no regex. Works for strings or Path-like objects.
        """
        if not isinstance(p, (str, Path)):
            return p
        s = str(p)
        parts = s.split('/')  # posix-style; your dataset paths use '/'
        for i, comp in enumerate(parts):
            if comp.endswith('.zip'):
                parts[i] = comp[:-4]  # strip ".zip"
        return '/'.join(parts)

    def load(self):
        loaded_data = []
        for example in self.dataset:
            img = example.get("image", None)
            if isinstance(img, (str, Path)):
                img = self._drop_zip_dir_component(img)
                    
            if self.direct_answer:
                if example["difficult_direct_answer"] and self.split in ["validation", "test"]:
                    continue

                out = dict(
                    image=img,
                    question=example["question"],
                    answers=example["direct_answers"],
                    metadata=dict(
                        example_id=example["question_id"]
                    )
                )
            else:
                if example["correct_choice_idx"] is None:
                    out = dict(
                        image=img,
                        question=example["question"],
                        options=example["choices"],
                        metadata=dict(example_id=example["question_id"])
                    )
                else:
                    out = dict(
                        image=img,
                        question=example["question"],
                        options=example["choices"],
                        answer_idx=example["correct_choice_idx"],
                        metadata=dict(example_id=example["question_id"])
                    )
            loaded_data.append(out)
        return loaded_data

    def __len__(self):
        return len(self.loaded_data)

    def get(self, item, rng):
        return dict(**self.loaded_data[item], style=self.style)
    
    

class OkVqa(Dataset):
    """
    OK-VQA dataset from HuggingFace M4 project.
    This class loads the OK-VQA dataset from HuggingFace (https://huggingface.co/datasets/HuggingFaceM4/OK-VQA).

    Args:
        split (str): Dataset split to load. One of "train", "validation", or "test".
        multi_question (bool, optional): Whether to group questions by image. Defaults to False.
    """

    PATH = "HuggingFaceM4/OK-VQA"

    @classmethod
    def download(cls, n_procs=1):
        datasets.load_dataset_builder(cls.PATH, trust_remote_code=True).download_and_prepare()

    def __init__(self, split: str, multi_question=False, keep_in_memory=False):
        super().__init__()
        self.multi_question = multi_question
        dataset = datasets.load_dataset(self.PATH, split=split, trust_remote_code=True, keep_in_memory=keep_in_memory)
        if self.multi_question:
            grouped_by_image = defaultdict(list)
            for ex in dataset:
                grouped_by_image[ex["image_id"]].append(ex)
            data = []
            for image_id, examples in grouped_by_image.items():
                questions = []
                for ex in examples:
                    questions.append(dict(
                        question=ex["question"],
                        answers=[x["raw_answer"] for x in ex["answers"]],
                    ))
                data.append(dict(
                    image=examples[0]["image"],
                    metadata=dict(image_id=image_id),
                    message_list=questions
                ))
            self.data = data
        else:
            self.data = dataset

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        if self.multi_question:
            return dict(ex, style="okvqa")
        else:
            return dict(
                image=ex["image"],
                question=ex["question"],
                answers=[x["raw_answer"] for x in ex["answers"]],
                metadata=dict(
                    example_id=ex["question_id"],
                ),
                style="okvqa",
            )


# class TextVqa(HfDataset):
#     """
#     This class loads the TextVQA dataset from HuggingFace (https://huggingface.co/datasets/facebook/textvqa).
#     """
#     PATH = "facebook/textvqa"

#     @classmethod
#     def download(cls, n_procs=1):
#         datasets.load_dataset_builder(cls.PATH, trust_remote_code=True).download_and_prepare()
#         # datasets.load_dataset(cls.PATH, trust_remote_code=True, streaming=True)# .download_and_prepare()
#     def __init__(self, split: str, identifier=None, keep_in_memory=False):
#         super().__init__(
#             split=split, keep_in_memory=keep_in_memory, trust_remote_code=True)

#     def get(self, item, rng):
#         example = self.dataset[item]
#         return dict(
#             image=example["image"],
#             question=example["question"],
#             answers=example.get("answers", []),
#             metadata=dict(
#                 image_url=example["flickr_300k_url"],
#                 image_id=example["image_id"],
#                 example_id=example["question_id"],
#             ),
#             style="text_vqa"
#         )




class TextVqa(HfDataset):
    PATH = "facebook/textvqa"

    @classmethod
    def download(cls, n_procs: int = 1):
        datasets.load_dataset_builder(cls.PATH, trust_remote_code=True).download_and_prepare()

    def __init__(
        self,
        split: str = "train",
        keep_in_memory: bool = False,
        skip_missing: bool = True,
        num_proc: Optional[int] = 12,
    ):
        ds = datasets.load_dataset(
            self.PATH,
            split=split,
            keep_in_memory=keep_in_memory,
            trust_remote_code=True,
        )

        # 🔑 Disable decoding BEFORE any access
        if "image" in ds.column_names:
            ds = ds.cast_column("image", Image(decode=False))

        # 🔑 Make sure 'image' won't open files when indexed
        if "image" in ds.column_names:
            ds = ds.cast_column("image", datasets.Image(decode=False))

        # Normalize image -> image_path (string only)
        if "image" in ds.column_names:
            def _to_path(ex):
                img = ex["image"]
                # when decode=False, img is usually {"path": "...", "bytes": None}
                if isinstance(img, dict) and "path" in img:
                    return {"image_path": img["path"]}
                return {"image_path": str(img) if img else None}
            ds = ds.map(_to_path, remove_columns=["image"])

        # Optionally drop rows pointing to non-existent files
        if skip_missing and "image_path" in ds.column_names:
            ds = ds.filter(
                lambda ex: isinstance(ex["image_path"], str) and os.path.exists(ex["image_path"]),
                num_proc=num_proc
            )

        self.dataset = ds

    def get(self, item, rng=None):
        ex = self.dataset[item]
        return dict(
            image=ex.get("image"),  # never auto-decoded
            question=ex.get("question"),
            answers=ex.get("answers", []),
            metadata=dict(
                image_url=ex.get("flickr_300k_url"),
                image_id=ex.get("image_id"),
                example_id=ex.get("question_id"),
                image_path=ex.get("image_path"),
            ),
            style="text_vqa",
        )

# class TextVqa(HfDataset):
#     """
#     This class loads the TextVQA dataset from HuggingFace (https://huggingface.co/datasets/facebook/textvqa).
#     """
#     PATH = "facebook/textvqa"

#     @classmethod
#     def download(cls, n_procs=1):
#         pass

#     # def __init__(self, split: str, identifier=None, keep_in_memory=False):
#     #     super().__init__(split=split, keep_in_memory=keep_in_memory, trust_remote_code=True)

#     def __init__(self, split: str, identifier=None, keep_in_memory=False):
#         self.dataset = load_dataset(
#             self.PATH,
#             split=split,
#             streaming=True,
#             trust_remote_code=True
#         )


#     def get(self, item, rng):
#         example = self.dataset[item]
#         return dict(
#             image=example["image"],
#             question=example["question"],
#             answers=example.get("answers", []),
#             metadata=dict(
#                 image_url=example["flickr_300k_url"],
#                 image_id=example["image_id"],
#                 example_id=example["question_id"],
#             ),
#             style="text_vqa"
#         )


class TallyQa(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        TallyQaBuilder().download_and_prepare()

    def __init__(self, split):
        assert split in ["train", "test"]
        self.dataset = TallyQaBuilder().as_dataset(split=split)
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        ex = self.dataset[item]
        messages = []
        questions = ex["questions"]
        for ix, question in enumerate(questions["question"]):
            messages.append(dict(
                question=question,
                answer=str(questions["answer"][ix]),
                style="tally_qa"
            ))
        return dict(
            image=ex["image"],
            message_list=messages,
            metadata=dict( image_id=ex["image_id"])
        )


class AI2D(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        Ai2dDatasetBuilder().download_and_prepare()

    def __init__(self, split, boxes="both"):
        assert split in ["train", "validation", "test"]
        dataset = Ai2dDatasetBuilder().as_dataset(split)
        if boxes == "transparent":
            dataset = dataset.filter(lambda x: not x["abc_label"] or x["has_transparent_box"])
        elif boxes == "opaque":
            dataset = dataset.filter(lambda x: not x["abc_label"] or not x["has_transparent_box"])
        elif boxes == "both":
            pass
        else:
            raise NotImplementedError(boxes)
        self.dataset = dataset

        self.split = split
        self.boxes = boxes
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        _ex = dict(self.dataset[item])
        ex = dict(
            image=_ex["image"],
            question=_ex["question"],
            answer_idx=_ex["correct_answer"],
            metadata=dict(
                example_id=_ex["question_id"],
                image_id=_ex["image_id"],
                abc_label=_ex["abc_label"],
                has_transparent_box=_ex["has_transparent_box"]
            ),
        )
        options = _ex["answer_texts"]
        if _ex["abc_label"] and sum(_ex["option_is_abc"]) >= (len(options)-1):
            ex["unlabelled_options"] = [
                opt.upper() if abc else opt
                for opt, abc in zip(options, _ex["option_is_abc"])
            ]
            ex["style"] = "ai2_diagram_no_letter"
        else:
            ex["options"] = options
            ex["style"] = "ai2_diagram"
        return ex


class ScienceQAImageOnly(Dataset):
    """
    This class loads the ScienceQA dataset from HuggingFace (https://huggingface.co/datasets/derek-thomas/ScienceQA).
    Merges all splits (train, validation, test) into a single dataset.
    """
    PATH = "derek-thomas/ScienceQA"

    @classmethod
    def download(self, n_procs=1):
        datasets.load_dataset_builder(self.PATH).download_and_prepare()

    def __init__(self, split):
        assert split in ["train", "validation", "test"]
        
        if split == "train":
            # Load all splits and merge them
            train_data = datasets.load_dataset(self.PATH, split="train").filter(lambda ex: ex["image"] is not None)
            val_data = datasets.load_dataset(self.PATH, split="validation").filter(lambda ex: ex["image"] is not None)
            test_data = datasets.load_dataset(self.PATH, split="test").filter(lambda ex: ex["image"] is not None)
            
            # Concatenate all splits
            from datasets import concatenate_datasets
            self.dataset = concatenate_datasets([train_data, val_data, test_data])
        else:
            # For validation/test splits, keep them as is (or you can make them empty)
            self.dataset = datasets.load_dataset(self.PATH, split=split).filter(lambda ex: ex["image"] is not None)
        
        super().__init__()

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        ex = self.dataset[item]
        question = ex["question"]
        hint = ex["hint"]
        if hint:
            question = hint + "\n" + question  # Fixed: was hint + "\n" + hint
        return dict(
            image=ex["image"],
            question=question,
            style="science_qa",
            answer_idx=ex["answer"],
            options=ex["choices"],
        )
        
        
class InfoQa(DatasetBase):
    SPLITS = ["train", "validation", "test"]

    @classmethod
    def download(cls, n_procs=1):
        for split in cls.SPLITS:
            if split == "validation":
                filename = "infographicsVQA_val_v1.0_withQT.json"
            else:
                filename = f"infographicsVQA_{split}_v1.0.json"
            if not exists(join(INFOQA_SOURCE, filename)):
                raise ValueError(
                    "InfoQa requires manually downloading https://rrc.cvc.uab.es/?ch=17 (Task 3)"
                    f" please download and unzip the data into `{INFOQA_SOURCE}`"
                )

    def __init__(self, split):
        assert split in self.SPLITS
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            filename = "infographicsVQA_val_v1.0_withQT.json"
        else:
            filename = f"infographicsVQA_{split}_v1.0.json"
        filename = join(INFOQA_SOURCE, filename)
        logging.info(f"Loading docqa data from {filename}")
        with open(filename) as f:
            data = json.load(f)
        out = []
        for ex in data["data"]:
            image_path = join(INFOQA_SOURCE, "infographicsvqa_images", ex.pop("image_local_name"))
            out.append(dict(
                image=image_path,
                question=ex["question"],
                answers=ex.get("answers", []),
                metadata=dict(example_id=ex["questionId"]),
            ))
        return out

    def get(self, item, rng):
        return dict(**self.data[item], style="info_qa")



class DocQa(HfDataset):
    """
    DocumentVQA dataset from HuggingFace M4 project.
    This class loads the DocumentVQA dataset from HuggingFace (https://huggingface.co/datasets/HuggingFaceM4/DocumentVQA).
    The dataset contains document images paired with questions and answers for visual document understanding tasks.

    Args:
        split (str): Dataset split to load. One of "train", "validation", or "test".
    """
    PATH = "HuggingFaceM4/DocumentVQA"

    def __init__(self, split: str, keep_in_memory=False, **kwargs):
        super().__init__(split, keep_in_memory, **kwargs)

    def get(self, item, rng):
        example = self.dataset[item]
        if self.split == "test":
            for k in ["answers", "question_types"]:
                assert k not in example or example[k] is None
                example[k] = []
        return dict(
                dict(
                image=example["image"],
                question=example["question"],
                answers=example.get("answers"),
                metadata=dict(
                    doc_id=example["docId"],
                    question_types=example.get("question_types"),
                    example_id=example["questionId"],
                )
            ), style="doc_qa")

        

class SceneTextQa(DatasetBase):

    @classmethod
    def download(cls, n_procs=1):
        for split in ["train", "test"]:
            if not exists(join(join(ST_QA_SRC, f"{split}_task_3.json"))):
                raise ValueError(
                    "SceneTextQa requires manually downloading https://rrc.cvc.uab.es/?ch=11"
                    f" please download and unzip the data into `{ST_QA_SRC}`"
                )

    def __init__(self, split):
        assert split in ["train", "test", "validation"]
        super().__init__(split)

    def load(self):
        split = self.split
        if split == "validation":
            split = "train"
        src = join(ST_QA_SRC, f"{self.split}_task_3.json")
        logging.info(f"Loading scene text data from {src}")
        with open(src) as f:
            data = json.load(f)["data"]
        out = []
        for question in data:
            out.append(dict(
                image=join(ST_QA_SRC, question["file_path"]),
                question=question["question"],
                metadata=dict(example_id=question["question_id"]),
                answers=question.get("answers", []),
            ))
        if self.split in ["train", "validation"]:
            # Custom val split since the data doesn't have one
            out.sort(key=lambda x: x["metadata"]["example_id"])
            np.random.RandomState(63069).shuffle(out)
            if self.split == "train":
                return out[1024:]
            else:
                return out[:1024]
        else:
            return out

    def get(self, item, rng):
        return dict(self.data[item], style="st_qa")




# class CountBenchQa(Dataset):
#     """
#     Dataset loader for CountBench QA dataset.
#     Loads byte-encoded images and count questions from Parquet files.
    
#     Dataset structure:
#         - image: dict with 'bytes' key containing raw image bytes
#         - text: descriptive text about the image
#         - question: the counting question
#         - number: the ground truth count
#     """

#     @classmethod
#     def download(cls, *args, **kwargs):
#         raise AssertionError("CountBench dataset must be pre-downloaded. No automatic download available.")

#     def __init__(self, split: str = "train", data_dir: str = "/Document/molmo_code/data/molmo/CountBench"):
#         """
#         Args:
#             split: Dataset split (train/validation/test)
#             data_dir: Root directory containing parquet files
#         """
#         assert split in ["train", "validation", "test"], f"Invalid split: {split}"
#         self.split = split
#         self.data_dir = data_dir
#         self._load_dataset()

#     def _load_dataset(self):
#         """Load all parquet files from the data directory"""
#         # Look for split-specific files first (e.g., train-*.parquet)
#         prefix = f"{self.split}-"
#         pattern = os.path.join(self.data_dir, f"{prefix}*.parquet")
#         parquet_files = sorted(glob(pattern))
        
#         # If no split-specific files, load all parquet files
#         if not parquet_files:
#             pattern = os.path.join(self.data_dir, "*.parquet")
#             parquet_files = sorted(glob(pattern))
        
#         if not parquet_files:
#             raise FileNotFoundError(
#                 f"No parquet files found for split '{self.split}' in directory: {self.data_dir}"
#             )

#         # Load and concatenate all parquet files
#         df_list = [pd.read_parquet(pf) for pf in parquet_files]
#         self.data = pd.concat(df_list, ignore_index=True)
        
#         print(f"Loaded {len(self.data)} samples from CountBench {self.split} split")

#     def __len__(self):
#         return len(self.data)

#     def get(self, item, rng=None):
#         """
#         Get a single sample from the dataset
        
#         Args:
#             item: Index of the sample
#             rng: Random number generator (optional, unused for this dataset)
            
#         Returns:
#             Dictionary with image, question, answers, metadata, and style
#         """
#         row = self.data.iloc[item]
        
#         # Extract raw bytes from image dict
#         raw_bytes = row['image']['bytes']
#         image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        
#         # Get the question directly from the dataset
#         question = row["question"]
        
#         # Get the ground truth count
#         count = int(row["number"])
#         answer = str(count)
        
#         # Generate image_id
#         image_id = f"countbench_{item}"
        
#         metadata = {
#             "count": count,
#             "image_id": image_id,
#             "image_url": "",
#             "text": row.get("text", "")  # Include descriptive text if needed
#         }
        
#         return dict(
#             image=image,
#             question=question,
#             answers=[answer],
#             metadata=metadata,
#             style="point_count"
#         )

import os, io
from glob import glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class CountBenchQa(Dataset):
    """
    Dataset loader for CountBench QA dataset.
    Loads byte-encoded images and count questions from Parquet files.

    Expected columns per row:
      - image: dict with key 'bytes' containing raw image bytes (or raw bytes directly)
      - text: optional descriptive text
      - question: the counting question (string)
      - number: the ground-truth count (int)
    """

    @classmethod
    def download(cls, *args, **kwargs):
        raise AssertionError("CountBench dataset must be pre-downloaded. No automatic download available.")

    def __init__(self, split: str = "train",
                 data_dir: str = "/Document/molmo_code/data/molmo/CountBench"):
        assert split in ["train", "validation", "test"], f"Invalid split: {split}"
        self.split = split
        self.data_dir = data_dir
        self._load_dataset()

    def _load_dataset(self):
        """Load all parquet files from the data directory"""
        prefix = f"{self.split}-"
        pattern = os.path.join(self.data_dir, f"{prefix}*.parquet")
        parquet_files = sorted(glob(pattern))

        if not parquet_files:
            # fall back to any parquet files
            parquet_files = sorted(glob(os.path.join(self.data_dir, "*.parquet")))

        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found for split '{self.split}' in directory: {self.data_dir}"
            )

        df_list = [pd.read_parquet(pf) for pf in parquet_files]
        self.data = pd.concat(df_list, ignore_index=True)
        print(f"Loaded {len(self.data)} samples from CountBench {self.split} split")

    def __len__(self):
        return len(self.data)

    # >>> This is what your DataLoader is calling
    def __getitem__(self, index):
        return self.get(index)

    def _decode_image(self, img_field):
        """
        Accepts:
          - dict-like with 'bytes' key
          - raw bytes / bytearray / memoryview
        Returns a PIL.Image in RGB.
        """
        if isinstance(img_field, dict) and "bytes" in img_field:
            raw = img_field["bytes"]
        else:
            raw = img_field

        if isinstance(raw, memoryview):
            raw = raw.tobytes()
        elif isinstance(raw, bytearray):
            raw = bytes(raw)

        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError(f"Unsupported image payload type: {type(raw)}")

        return Image.open(io.BytesIO(raw)).convert("RGB")

    def get(self, item, rng=None):
        """
        Returns:
          dict(image: PIL.Image, question: str, answers: [str], metadata: dict, style: str)
        """
        row = self.data.iloc[item]

        # Decode image
        image = self._decode_image(row["image"])

        # Question and answer
        question = row["question"]
        count = int(row["number"])
        answer = str(count)

        # Metadata
        image_id = f"countbench_{item}"
        meta_text = row["text"] if "text" in row else (row.get("text", "") if hasattr(row, "get") else "")

        metadata = {
            "count": count,
            "image_id": image_id,
            "image_url": "",
            "text": meta_text,
        }

        return dict(
            image=image,
            question=question,
            answers=[answer],
            metadata=metadata,
            style="point_count",
        )




class TabWMPDirectAnswer(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        TabMwpBuilder().download_and_prepare()

    def __init__(self, split, include_options: bool):
        self.include_options = include_options
        self._dataset = TabMwpBuilder().as_dataset(split)

    def __len__(self):
        return len(self._dataset)

    def get(self, item, rng):
        ex = self._dataset[item]
        out = dict(
            image=ex["image"],
            question=ex["question"],
            answer=ex["answer"],
            style="tabwmp_da",
            metadata=dict(
                example_id=ex["example_id"]
            )
        )
        if self.include_options and ex["choices"]:
            out["options"] = ex["choices"]
        return out


class FigureQa(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        FigureQaBuilder().download_and_prepare()

    def __init__(self, split, in_memory=False):
        assert split in ["train", "validation1", "test1", "validation2", "test2"]
        self.hf_dataset = FigureQaBuilder().as_dataset(split, in_memory=in_memory)

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        qas = example["questions"]
        messages = []
        for q, a in zip(qas["question"], qas["answer"]):
            messages.append(dict(question=q, answer=str(a), style="figure_qa"))
        return dict(image=example["image"], message_list=messages)

    def __len__(self):
        return len(self.hf_dataset)


class PlotQa(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        PlotQaBuilder().download_and_prepare()

    def __init__(self, split, in_memory=False):
        assert split in ["train", "validation", "test"]
        self.hf_dataset = PlotQaBuilder().as_dataset(split, in_memory=in_memory)

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        qas = example["questions"]
        messages = []
        for q, a in zip(qas["question"], qas["answer"]):
            messages.append(dict(question=q, answer=a, style="plot_qa"))
        return dict(image=example["image"], message_list=messages)

    def __len__(self):
        return len(self.hf_dataset)


class AndroidControl(Dataset):
    @classmethod
    def download(cls, n_procs=1):
        AndroidControlBuilder().download_and_prepare(num_proc=n_procs)

    def __init__(self, split, mode="all", in_memory=False):
        self.mode = mode
        self.hf_dataset = AndroidControlBuilder().as_dataset(
            "val" if split == "validation" else split, in_memory=in_memory)

    def __len__(self):
        return len(self.hf_dataset)

    def get(self, item, rng):
        ex = self.hf_dataset[item]
        ll, hl_ll, hl, hl_cot = [
            dict(
                prompt="low_level: " + ex["ll_instruction"],
                text=ex["target_action"],
                style="android_control"
            ),
            dict(
                prompt="high_level: " + ex["hl_instruction"] + " low_level: " + ex["ll_instruction"],
                text=ex["target_action"],
                style="android_control"
            ),
            dict(
                prompt="high_level: " + ex["hl_instruction"],
                text=ex["target_action"],
                style="android_control"
            ),
            dict(
                prompt="high_level_cot: " + ex["hl_instruction"],
                text="Plan: " + ex["ll_instruction"] + " Action: " + ex["target_action"],
                style="android_control"
            )
        ]
        example = dict(
            image=ex["image"],
            metadata=dict(
                target_action=ex["target_action"],
                target_box=ex["target_box"],
                ll_instruction=ex["ll_instruction"],
                hl_instruction=ex["hl_instruction"],
            )
        )
        if self.mode == "ll":
            example.update(ll)
        elif self.mode == "hl":
            example.update(hl)
        elif self.mode == "hl_ll":
            example.update(hl_ll)
        elif self.mode == "hl_cot":
            example.update(hl_cot)
        elif self.mode == "all":
            example["message_list"] = [ll, hl_ll, hl, hl_cot]
        else:
            raise NotImplementedError(self.mode)
        return example


class DvQa(Dataset):
    @classmethod
    def download(cls, n_procs=1):
        DvQaBuilder().download_and_prepare()

    def __init__(self, split, in_memory=False):
        self.hf_dataset = DvQaBuilder().as_dataset(split, in_memory=in_memory)

    def __len__(self):
        return len(self.hf_dataset)

    def get(self, item, rng):
        example = self.hf_dataset[int(item)]
        qas = example["questions"]
        messages = []
        for q, a in zip(qas["question"], qas["answer"]):
            messages.append(dict(question=q, answer=a, style="dv_qa"))
        return dict(
            image=example["image"],
            message_list=messages,
            metadata=dict(image_id=example["image_id"]),
        )


# class MathVista(HfDataset):
#     PATH = "AI4Math/MathVista"

#     def __init__(self, split, simplify_question=True, **kwargs):
#         super().__init__(split, **kwargs)
#         self.simplify_question = simplify_question

#     def get(self, item, rng):
#         ex = self.dataset[item]
#         question: str = ex["question"]
#         if self.simplify_question:
#             question = question.split("Question:")[-1]
#             question = question.split("Hint:")[0].strip()
#         out = dict(
#             question=question,
#             image=ex["decoded_image"],
#             metadata=dict(
#                 example_id=ex["pid"],
#                 answer=ex["answer"],
#                 precision=ex["precision"],
#                 query=ex["question"],
#                 choices=ex["choices"],
#                 question_type=ex["question_type"],
#                 answer_type=ex["answer_type"]
#             ),
#         )
#         if ex["question_type"] == "multi_choice":
#             out["options"] = ex["choices"]
#             out["style"] = "eval_multiple_choice"
#         else:
#             out["style"] = "eval_short_answer"
#         return out



class MathVista(Dataset):
    """
    Dataset loader for MathVista dataset.
    Loads byte-encoded images and math questions from Parquet files.
    
    Dataset structure:
        - pid: problem ID
        - question: the math question
        - image: image path (unused)
        - decoded_image: dict with 'bytes' key containing raw image bytes
        - choices: multiple choice options (None for free_form)
        - answer: ground truth answer
        - question_type: 'multi_choice' or 'free_form'
        - answer_type: type of answer (integer, float, text, etc.)
    """

    @classmethod
    def download(cls, *args, **kwargs):
        raise AssertionError("MathVista dataset must be pre-downloaded. No automatic download available.")

    def __init__(self, split: str = "train", data_dir: str = "/Document/molmo_code/data/molmo/MathVista", simplify_question: bool = True):
        """
        Args:
            split: Dataset split (train/validation/test)
            data_dir: Root directory containing parquet files
            simplify_question: Whether to simplify questions by removing "Question:" and "Hint:" prefixes
        """
        assert split in ["train", "validation", "test"], f"Invalid split: {split}"
        self.split = split
        self.data_dir = data_dir
        self.simplify_question = simplify_question
        self._load_dataset()

    def _load_dataset(self):
        """Load all parquet files from the data directory"""
        # Look for split-specific files first (e.g., train-*.parquet)
        prefix = f"{self.split}-"
        pattern = os.path.join(self.data_dir, f"{prefix}*.parquet")
        parquet_files = sorted(glob(pattern))
        
        # If no split-specific files, load all parquet files
        if not parquet_files:
            pattern = os.path.join(self.data_dir, "*.parquet")
            parquet_files = sorted(glob(pattern))
        
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found for split '{self.split}' in directory: {self.data_dir}"
            )

        # Load and concatenate all parquet files
        df_list = [pd.read_parquet(pf) for pf in parquet_files]
        self.data = pd.concat(df_list, ignore_index=True)
        
        print(f"Loaded {len(self.data)} samples from MathVista {self.split} split")

    def __len__(self):
        return len(self.data)

    def get(self, item, rng=None):
        """
        Get a single sample from the dataset
        
        Args:
            item: Index of the sample
            rng: Random number generator (optional, unused for this dataset)
            
        Returns:
            Dictionary with image, question, answers, metadata, and style
        """
        row = self.data.iloc[item]
        
        # Extract raw bytes from decoded_image dict
        raw_bytes = row['decoded_image']['bytes']
        image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        
        # Get and optionally simplify the question
        question = row["question"]
        if self.simplify_question:
            question = question.split("Question:")[-1]
            question = question.split("Hint:")[0].strip()
        
        # Build metadata
        metadata = {
            "example_id": row["pid"],
            "answer": row["answer"],
            "precision": row.get("precision"),
            "query": row.get("query", row["question"]),
            "choices": row.get("choices"),
            "question_type": row["question_type"],
            "answer_type": row["answer_type"],
            "unit": row.get("unit"),
        }
        
        # Add additional metadata if present
        if "metadata" in row and pd.notna(row["metadata"]):
            if isinstance(row["metadata"], dict):
                metadata.update(row["metadata"])
        
        # Build output dictionary
        out = {
            "image": image,
            "question": question,
            "answers": [str(row["answer"])],
            "metadata": metadata,
        }
        
        # Set style and options based on question type
        if row["question_type"] == "multi_choice":
            out["options"] = row["choices"]
            out["style"] = "eval_multiple_choice"
        else:
            out["style"] = "eval_short_answer"
        
        return out
    
    
    
class RealWorldQa(HfDataset):
    PATH = "xai-org/RealworldQA"

    def __init__(self, mode="no_mc_instruction", in_memory=False):
        super().__init__("test", in_memory)
        self.mode = mode

    def get(self, item, rng):
        ex = self.dataset[item]
        prompt: str = ex["question"]
        if "Please answer directly with a single word or number." in prompt:
            question_type = "short_answer"
        else:
            assert "Please answer directly with only the letter of the correct option and nothing else." in prompt
            question_type = "multiple_choice"
        out = dict(
            image=ex["image"],
            metadata=dict(answer=ex["answer"], prompt=ex["question"], question_type=question_type),
        )
        if self.mode == "plain":
            out.update(style="none", prompt=prompt)
        else:
            if question_type == "short_answer":
                style = "eval_short_answer"
            else:
                style = "eval_multiple_choice"
            if self.mode == "no_instruction":
                if question_type == "short_answer":
                    prompt = prompt.split("\n")[0]
            else:
                if self.mode != "vqa_style_tag":
                    raise NotImplementedError(self.mode)
            out.update(style=style, question=prompt)
        return out


class MMMU(Dataset):
    NAMES = [
        'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory',
        'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science',
        'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power',
        'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math',
        'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health',
        'Sociology'
    ]

    @classmethod
    def download(cls, n_procs=1):
        for name in cls.NAMES:
            if exists(join(DATA_HOME, "mmmu", name)):
                continue
            builder = datasets.load_dataset_builder("MMMU/MMMU", name=name)
            builder.download_and_prepare()

    def __init__(self, split: str):
        all_parts = []
        for name in self.NAMES:
            all_parts.append(datasets.load_dataset("MMMU/MMMU", name=name, split=split))
        self.data = datasets.concatenate_datasets(all_parts)

    def __len__(self):
        return len(self.data)

    def get(self, item, rng):
        ex = self.data[item]
        mc = ex["question_type"] == "multiple-choice"
        out = dict(
            image=ex["image_1"],
            text=ex["answer"],
            question=ex["question"],
            metadata=dict(answer=ex["answer"], example_id=ex["id"], question_type=ex["question_type"]),
            style='a_okvqa_mc' if mc else 'vqa2'
        )
        if mc:
            options = eval(ex["options"])
            if sum((re.match("<img='(.*?)'>", opt) is not None) for opt in options) > 1:
                # Following LLaVa, don't use any images if there are multiple images paths
                # I think the rationale is that this means the image are answer-options
                del out["image"]
            out["options"] = options
        return out


class ClockBench(Dataset):

    @classmethod
    def download(cls, n_procs=1):
        ClockBenchBuilder().download_and_prepare()

    def __init__(self, split):
        assert split in ["coco", "openimg", "movies"]
        dataset = ClockBenchBuilder().as_dataset(split)
        self.dataset = dataset
        self.split = split

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        _ex = dict(self.dataset[item])
        hour, minute = [int(_ex[k]) for k in ["hour", "minute"]]
        if hour == 12:
            hour = 0
        second = -1
        return dict(
            image=np.array(_ex["image"]),
            prompt="What time is being shown?",
            metadata=dict(
                hour=hour,
                minute=minute,
                second=second,
                example_id=_ex["image_id"],
            ),
            style="clocks",
        )




## Bounding boxes

class COCO_BBOX(Dataset):
    """
    Dataloader for JSONL 'separated' COCO bbox data.

    Each JSONL record looks like:
      {
        "id": "...",
        "image": ["images/coco/coco_9.jpg"],
        "conversations": [
          {"role":"user","content":[{"type":"text","text":"..."},{"type":"image","text":"<image>"}]},
          {"role":"assistant","content":[
              {"type":"bbox","bbox":[[[x1%,y1%],[x2%,y2%]], ...], "label":"..."},
              {"type":"text","text":"So the number of ... is: K\n"}
          ]}
        ],
        ...
      }

    Returns dict with:
      image: PIL.Image (RGB)
      question: str  (instruction + user question)
      answers: [str] (assistant text + "<box> [...] </box>")
      metadata: dict (id, image_path, label, pixel_boxes, percent_boxes, etc.)
      style: "text_vqa"
    """

    IMAGE_ROOT = "/Document/molmo_code/data/molmo/COCO_Bbox/images"
    # JSONL_PATHS = {
    #     "train": "/Document/molmo_code/data/molmo/COCO_Bbox/bbox-coco-0418_train.jsonl",
    #     "validation": "/Document/molmo_code/data/molmo/COCO_Bbox/bbox-coco-0418_test.jsonl",
    #     "test": "/Document/molmo_code/data/molmo/COCO_Bbox/bbox-coco-0418_test.jsonl",
    # }
    JSONL_PATHS = {
        "train": "/Document/molmo_code/data/molmo/COCO_Bbox/bbox-coco-0418_train.jsonl",
        "validation": "/Document/molmo_code/data/molmo/COCO_Bbox/bbox-coco-0418_train_first500.jsonl",
        "test": "/Document/molmo_code/data/molmo/COCO_Bbox/bbox-coco-0418_train_first500.jsonl",
    }

    INSTRUCTION = "## Read the question, give the answer and the <box> [x1, y1, x2, y2], [...] </box>.\n"

    def __init__(
        self,
        split: str = "train",
        image_size: Optional[Tuple[int, int]] = None,
        image_root: Optional[str] = None,
        jsonl_path: Optional[str] = None,
        drop_missing_images: bool = True,
        max_records: Optional[int] = None,
    ):
        assert split in ["train", "validation", "test"], f"Invalid split: {split}"

        self.split = split
        self.image_size = image_size  # (W, H) or None (keep original)
        self.image_root = image_root or self.IMAGE_ROOT
        self.jsonl_path = jsonl_path or self.JSONL_PATHS[split]
        self.drop_missing_images = drop_missing_images

        self.records: List[Dict[str, Any]] = []
        self._load_jsonl(max_records=max_records)

    # --------------- utils ---------------

    @staticmethod
    def _first(lst, default=None):
        return lst[0] if isinstance(lst, list) and lst else default

    def _resolve_image_path(self, rel_or_abs: str) -> Path:
        """Resolve absolute path; tolerate 'images/...' inside rel when IMAGE_ROOT ends with 'images'."""
        p = Path(rel_or_abs)
        if p.is_absolute():
            return p
        base = Path(self.image_root)
        if base.name == "images" and p.parts and p.parts[0] == "images":
            p = Path(*p.parts[1:])
        return base / p

    @staticmethod
    def _extract_user_question(conversations: List[Dict[str, Any]]) -> str:
        # last user turn text
        q = ""
        for turn in conversations:
            if turn.get("role") == "user":
                for c in turn.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "text":
                        q = c.get("text", "").strip()
        return q

    @staticmethod
    def _extract_assistant_bbox_and_text(conversations: List[Dict[str, Any]]):
        """
        Return (label, percent_boxes, assistant_text)
          label: str or None
          percent_boxes: List of [[x1%,y1%],[x2%,y2%]]
          assistant_text: str (may be '')
        """
        label, boxes, atext = None, [], ""
        for turn in conversations:
            if turn.get("role") == "assistant":
                cont = turn.get("content", [])
                for i, item in enumerate(cont):
                    if isinstance(item, dict) and item.get("type") == "bbox":
                        label = item.get("label", label)
                        # Keep full list-of-boxes
                        b = item.get("bbox", [])
                        if isinstance(b, list):
                            for bb in b:
                                if isinstance(bb, list) and len(bb) == 2:
                                    boxes.append(bb)
                        # Try to pick the nearest assistant text in same turn
                        for j in range(i + 1, len(cont)):
                            if isinstance(cont[j], dict) and cont[j].get("type") == "text":
                                atext = cont[j].get("text", "").strip()
                                break
        return label, boxes, atext

    @staticmethod
    def _percent_to_pixel(box_pct, W: int, H: int):
        """box_pct: [[x1%, y1%], [x2%, y2%]] → (x1,y1,x2,y2) pixel (float)."""
        (x1p, y1p), (x2p, y2p) = box_pct
        x1 = (x1p / 100.0) * W
        y1 = (y1p / 100.0) * H
        x2 = (x2p / 100.0) * W
        y2 = (y2p / 100.0) * H
        return x1, y1, x2, y2

    @staticmethod
    def _fmt_box_list_pixel(boxes_xyxy: List[Tuple[float, float, float, float]]) -> str:
        """
        Format a list of pixel boxes as: <box> [x1, y1, x2, y2], [...] </box>
        Integers are typically preferred for text outputs.
        """
        parts = []
        for (x1, y1, x2, y2) in boxes_xyxy:
            parts.append(f"[{int(round(x1))}, {int(round(y1))}, {int(round(x2))}, {int(round(y2))}]")
        return "<box> " + ", ".join(parts) + " </box>"

    # --------------- IO ---------------

    def _load_jsonl(self, max_records: Optional[int] = None):
        path = Path(self.jsonl_path)
        assert path.is_file(), f"JSONL not found: {path}"

        kept = 0
        with path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue

                # Resolve image path
                img_field = rec.get("image")
                rel = self._first(img_field, None) if img_field is not None else None
                if not isinstance(rel, str):
                    continue
                img_path = self._resolve_image_path(rel)

                # If drop_missing_images, skip; else keep (it may error later on get)
                if self.drop_missing_images and not img_path.exists():
                    continue

                # Minimal sanity: must have conversations and an assistant bbox block
                convs = rec.get("conversations", [])
                if not isinstance(convs, list) or not convs:
                    continue
                label, pct_boxes, atext = self._extract_assistant_bbox_and_text(convs)
                if not pct_boxes:
                    continue

                self.records.append(rec)
                kept += 1
                if max_records and kept >= max_records:
                    break

        if len(self.records) == 0:
            raise RuntimeError(
                f"No usable records loaded from {path}. "
                f"(drop_missing_images={self.drop_missing_images}, image_root={self.image_root})"
            )

    # --------------- dataset API ---------------

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, item, rng=None):
        rec = self.records[item]

        # Resolve image path & load
        rel = self._first(rec.get("image", []))
        img_path = self._resolve_image_path(rel)
        image = Image.open(img_path).convert("RGB")

        # Original size
        W0, H0 = image.size

        # Resize if requested
        if self.image_size is not None:
            # image_size = (W, H)
            Wt, Ht = self.image_size
            if (Wt, Ht) != (W0, H0):
                image = image.resize((Wt, Ht))
                W_img, H_img = Wt, Ht
            else:
                W_img, H_img = W0, H0
        else:
            W_img, H_img = W0, H0

        # Extract question (user text), assistant bbox + text
        convs = rec.get("conversations", [])
        question_user = self._extract_user_question(convs)
        label, pct_boxes, assistant_text = self._extract_assistant_bbox_and_text(convs)

        # Convert all percent boxes to pixel boxes in the (possibly resized) image space
        pixel_boxes = [self._percent_to_pixel(bb, W_img, H_img) for bb in pct_boxes]

        # Build prompt and answer as requested
        question = self.INSTRUCTION + (question_user or "Mark the objects.\n")

        # Assistant text may already have a trailing newline; trim and append boxes
        a_text = (assistant_text or "").strip()
        box_str = self._fmt_box_list_pixel(pixel_boxes)
        if a_text:
            answer = f"{a_text} and {box_str}"
        else:
            # If there was no explicit assistant text, synthesize a minimal answer:
            if label:
                answer = f"{label} {box_str}"
            else:
                answer = box_str

        metadata = {
            "id": rec.get("id"),
            "image_path": str(img_path),
            "label": label,
            "percent_boxes": pct_boxes,                       # original %-based boxes
            "pixel_boxes_xyxy": [list(map(int, map(round, b))) for b in pixel_boxes],
            "orig_size": [W0, H0],
            "final_size": [W_img, H_img],
        }

        
        return {
            "image": image,          # PIL.Image (RGB)
            "question": question,    # str
            "answers": [answer],     # list[str]
            "metadata": metadata,
            "style": "chart_qa",
        }

















from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class RefCoco(Dataset):
    """
    RefCOCO dataset loader for referring expression comprehension.
    Uses pre-verified clean parquet files that only contain valid rows.
    
    Returns bounding boxes in [x_min, y_min, x_max, y_max] format.
    """
    
    IMAGE_ROOT = "/Pixmo_Dataset/refcoco-dataset/train2014"
    DATA_ROOT = "/Pixmo_Dataset/refcoco-dataset/verified"
    DATA_PATHS = {
        "train": "train_clean.parquet",
        "validation": "validation_clean.parquet",
        "test": "test_clean.parquet",
        "testA": "test_clean.parquet",
        "testB": "testB_clean.parquet",
    }

    INSTRUCTION = "You are a vision-grounding expert. Read the referring caption carefully, then identify and localize the described object or region in the image. Output the bounding box coordinates in the format: [x_min, y_min, x_max, y_max] using precise pixel values."

    def __init__(
        self,
        split: str = "train",
        image_size: Optional[Tuple[int, int]] = None,
        image_root: Optional[str] = None,
        data_root: Optional[str] = None,
        max_records: Optional[int] = None,
    ):
        self.split = self._normalize_split(split)
        self.image_size = image_size
        self.image_root = Path(image_root or self.IMAGE_ROOT)
        self.data_root = Path(data_root or self.DATA_ROOT)
        
        # Load data (no need for drop_missing_images - clean data already verified)
        self.data = self._load_data(max_records=max_records)
        print(f"[RefCoco] Loaded {len(self.data)} verified examples for {self.split} split")

    def _normalize_split(self, split: str) -> str:
        """Normalize split names and handle aliases."""
        split_key = split.lower()
        aliases = {
            "val": "validation",
            "valid": "validation",
            "testa": "testA",
            "testb": "testB",
        }
        normalized = aliases.get(split_key, split)
        if normalized not in self.DATA_PATHS:
            raise ValueError(f"Invalid split: {split}. Valid splits: {list(self.DATA_PATHS.keys())}")
        return normalized

    def _resolve_image_path(self, image_path: str) -> Path:
        """Resolve image_path against image_root, keeping train2014/val2014 when present."""
        p = Path(image_path)
        if p.is_absolute() and p.exists():
            return p

        parts = p.parts
        rel = None
        for token in ("train2014", "val2014"):
            if token in parts:
                rel = Path(*parts[parts.index(token):])
                break
        if rel is None:
            rel = Path(p.name)

        base = self.image_root
        if rel.parts and base.name in ("train2014", "val2014") and base.name == rel.parts[0]:
            rel = Path(*rel.parts[1:])
        return base / rel

    def _load_data(self, max_records: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load data from the clean parquet file (already verified)."""
        
        # Load the clean parquet file for this split
        parquet_file = self.DATA_PATHS[self.split]
        path = self.data_root / parquet_file
        
        if not path.exists():
            raise FileNotFoundError(
                f"Clean parquet file not found: {path}\n"
                f"Please run the image availability checker first to generate clean parquet files."
            )
        
        df = pd.read_parquet(path)
        print(f"[RefCoco] Loaded {self.split} split: {len(df)} rows from {parquet_file}")
        
        # Process rows - all rows in clean files are already validated
        data = []
        
        for idx, row in df.iterrows():
            try:
                # Resolve image path
                full_image_path = self._resolve_image_path(row["image_path"])
                
                # Parse bbox - expecting format: [x, y, w, h]
                bbox = row["bbox"]
                if isinstance(bbox, str):
                    import json
                    bbox = json.loads(bbox)
                
                if len(bbox) == 4:
                    x, y, w, h = bbox
                else:
                    print(f"Warning: Unexpected bbox format at index {idx}: {bbox}")
                    continue
                
                # Convert to [x_min, y_min, x_max, y_max]
                x_min, y_min = x, y
                x_max, y_max = x + w, y + h
                
                # Get captions - handle both single caption and list of captions
                captions = row.get("captions", row.get("sentences", ""))
                if isinstance(captions, list):
                    caption = captions[0] if captions else ""
                else:
                    caption = str(captions)
                
                # Get image dimensions from raw_image_info if available
                raw_image_info = row.get("raw_image_info", {})
                if isinstance(raw_image_info, str):
                    import json
                    raw_image_info = json.loads(raw_image_info)
                
                width = raw_image_info.get("width") if isinstance(raw_image_info, dict) else None
                height = raw_image_info.get("height") if isinstance(raw_image_info, dict) else None
                
                # Calculate bbox area
                bbox_area = w * h
                
                data.append({
                    "id": row.get("ann_id", idx),
                    "caption": caption,
                    "bbox_xywh": [x, y, w, h],
                    "bbox_xyxy": [x_min, y_min, x_max, y_max],
                    "width": width,
                    "height": height,
                    "file_name": full_image_path.name,
                    "image_path": str(full_image_path),
                    "bbox_area": bbox_area,
                    "image_id": row.get("image_id"),
                    "ref_id": row.get("ref_id"),
                })
                
                if max_records and len(data) >= max_records:
                    break
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        if not data:
            raise RuntimeError(f"No valid data loaded for split {self.split}")
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, item, rng=None):
        row = self.data[item]
        
        # Load image
        image = Image.open(row["image_path"]).convert("RGB")
        W0, H0 = image.size
        
        # Update dimensions if not available in metadata
        if row["width"] is None:
            row["width"] = W0
        if row["height"] is None:
            row["height"] = H0
        
        # Optional resize
        if self.image_size is not None:
            Wt, Ht = self.image_size
            if (Wt, Ht) != (W0, H0):
                image = image.resize((Wt, Ht))
                W_img, H_img = Wt, Ht
            else:
                W_img, H_img = W0, H0
        else:
            W_img, H_img = W0, H0
        
        # Scale bbox if resized
        x_min, y_min, x_max, y_max = row["bbox_xyxy"]
        if (W_img, H_img) != (W0, H0):
            sx = W_img / W0
            sy = H_img / H0
            x_min, x_max = x_min * sx, x_max * sx
            y_min, y_max = y_min * sy, y_max * sy
        
        # Clip to image bounds
        x_min = max(0, min(W_img, x_min))
        y_min = max(0, min(H_img, y_min))
        x_max = max(0, min(W_img, x_max))
        y_max = max(0, min(H_img, y_max))
        
        # Ensure valid bbox
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        if y_max < y_min:
            y_min, y_max = y_max, y_min
        
        # Format question and answer
        question = self.INSTRUCTION + f" Caption: {row['caption']}\n"
        answer = f"[{int(round(x_min))}, {int(round(y_min))}, {int(round(x_max))}, {int(round(y_max))}]"
        
        return {
            "image": image,
            "question": question,
            "answers": [answer],
            "metadata": {
                "id": row["id"],
                "image_id": row["image_id"],
                "ref_id": row["ref_id"],
                "image_path": row["image_path"],
                "caption": row["caption"],
                "bbox_xywh": row["bbox_xywh"],
                "bbox_xyxy": [int(round(x_min)), int(round(y_min)), int(round(x_max)), int(round(y_max))],
                "orig_size": [W0, H0],
                "final_size": [W_img, H_img],
                "file_name": row["file_name"],
                "bbox_area": row["bbox_area"],
            },
            "style": "refcoco",
        }


