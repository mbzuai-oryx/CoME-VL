import csv
import logging
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from os.path import join, exists

import datasets
import numpy as np
from datasets import load_from_disk, concatenate_datasets, DatasetDict

from olmo.data.dataset import DATA_HOME, Dataset
from olmo.data.download_urls import download_pixmo_urls, filter_and_group_data

import os
from os.path import join, exists
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import ast

PIXMO_DATASETS = '/Pixmo_Dataset/'


# if DATA_HOME is not None:
#     PIXMO_DATASETS = join(DATA_HOME, "pixmo_datasets")
# else:
#     PIXMO_DATASETS = None
# """Where to save local version of the data after URLs filtering"""

PIXMO_LOCAL_DATASETS = os.environ.get(
    "PIXMO_LOCAL_DATASETS",
    "/Pixmo_Dataset",
)
"""Local PixMo dataset root with subfolders like pixmo-count, pixmo-docs, etc."""

VERIFY = True
"""Verify SSL certificates when downloading"""


NO_POINT_PREFIX = [
    "No pointing: ",
    "No pointing: ",
    "no pointing:\n",
    "No pointing:\n",
    "Not pointing:\n",
    "No Points: ",
    "No Points: ",
    "NO POINTING\n",
    "No pontiing\n",
    "No Points:\n ",
    "No pointing\n",
    "Do not point. ",
    "Refrain from pointing. ",
    "Avoid generating points . ",
    "For this question, do not use points. ",
    "Refrain from using points:\n",
    "Don't include points in your response. ",
    "Don't point. ",
    "Don't use points. ",
    "Please don't use points.\n\n",
    "Please don't use points.\n\n",
    "Respond without using points. ",
    "Respond without pointing:\n",
    "Do not generate ponits: ",
    "Do not point. ",
    "Do not point\n",
    "no pointing\n\n",
    "Answer without points: ",
    "Answer this question without pointing: ",
    "Answer without poiints. ",
    "answer without points: ",
    "answer with text only, do not points\n"
]
"""No-pointing requests templates, used for preprocessing"""

def _safe_text(value):
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _local_dataset_root() -> str:
    if PIXMO_LOCAL_DATASETS is None or not exists(PIXMO_LOCAL_DATASETS):
        raise FileNotFoundError(
            "PIXMO_LOCAL_DATASETS is not set or does not exist: "
            f"{PIXMO_LOCAL_DATASETS}"
        )
    return PIXMO_LOCAL_DATASETS


def _local_dataset_path(name: str) -> str:
    return join(_local_dataset_root(), name)


def _local_images_dir(name: str) -> str:
    return join(_local_dataset_path(name), "images")


def _image_path_from_sha(dataset_name: str, image_sha256: str) -> str:
    return join(_local_images_dir(dataset_name), f"{image_sha256}.jpg")


def _load_image_url_to_filename(csv_path: str) -> dict:
    # Increase CSV field size limit for large columns (e.g., masks or transcripts).
    max_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(max_limit)
            break
        except OverflowError:
            max_limit = max_limit // 10
    url_to_filename = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            url = row.get("image_url")
            filename = row.get("saved_filename")
            if url and filename:
                url_to_filename[url] = filename
    return url_to_filename


def _add_image_from_sha(
    dataset: datasets.Dataset, dataset_name: str, n_procs: int
) -> datasets.Dataset:
    base = _local_images_dir(dataset_name)

    def _add(batch):
        return {
            "image": [
                join(base, f"{sha}.jpg") if sha is not None else None
                for sha in batch["image_sha256"]
            ]
        }

    return dataset.map(_add, batched=True, num_proc=n_procs)


def _add_image_from_url_map(
    dataset: datasets.Dataset, dataset_name: str, url_to_filename: dict, n_procs: int
) -> datasets.Dataset:
    base = _local_images_dir(dataset_name)

    def _add(batch):
        images = []
        for url in batch["image_url"]:
            filename = url_to_filename.get(url)
            images.append(join(base, filename) if filename else None)
        return {"image": images}

    return dataset.map(_add, batched=True, num_proc=n_procs)


def _filter_existing_images(dataset: datasets.Dataset, n_procs: int) -> datasets.Dataset:
    def _keep(batch):
        return [isinstance(p, str) and exists(p) for p in batch["image"]]

    return dataset.filter(_keep, batched=True, num_proc=n_procs)


def _group_by_image_url_with_image(data: datasets.Dataset) -> datasets.Dataset:
    grouped_by_url = defaultdict(list)
    for example in data:
        grouped_by_url[example["image_url"]].append(example)

    grouped_examples = []
    for image_url, examples in grouped_by_url.items():
        grouped = dict(
            image_url=image_url,
            image=examples[0]["image"],
        )
        annotations = defaultdict(list)
        for ex in examples:
            for k, v in ex.items():
                if k not in ["image_url", "image_sha256", "image"]:
                    annotations[k].append(v)
        grouped.update(annotations)
        grouped_examples.append(grouped)
    return datasets.Dataset.from_list(grouped_examples)


def _select_split(dataset_or_dict, split: str):
    if isinstance(dataset_or_dict, DatasetDict):
        return dataset_or_dict[split]
    if split != "train":
        raise ValueError(f"Split {split} not available in a single-split dataset")
    return dataset_or_dict


def save_local_dataset(dataset: datasets.Dataset, name: str, n_procs, n_val=None):
    if len(dataset) == 0:
        raise ValueError("Given an empty dataset")
    if n_val:
        split = dataset.train_test_split(test_size=n_val, seed=96817)
        dataset = datasets.DatasetDict(train=split["train"], validation=split["test"])
    logging.info("Preparing local dataset...")
    if exists(name):
        logging.info(f"{name} already exists, it will be removed")
        shutil.rmtree(name)
    dataset.save_to_disk(name, num_proc=n_procs)
    logging.info("Done")


def _pixmo_cache_only() -> bool:
    return os.environ.get("PIXMO_CACHE_ONLY", "").lower() in ("1", "true", "yes")


def _is_dataset_dir(path: str) -> bool:
    if exists(join(path, "dataset_info.json")) or exists(join(path, "state.json")):
        return True
    if exists(join(path, "dataset_dict.json")):
        for split in ("train", "validation", "test"):
            if exists(join(path, split, "dataset_info.json")) or exists(join(path, split, "state.json")):
                return True
        return False
    for split in ("train", "validation", "test"):
        if exists(join(path, split, "dataset_info.json")) or exists(join(path, split, "state.json")):
            return True
    return False


def _has_required_splits(path: str, required_splits) -> bool:
    if required_splits is None:
        return _is_dataset_dir(path)
    if exists(join(path, "dataset_dict.json")):
        for split in required_splits:
            if not (
                exists(join(path, split, "dataset_info.json"))
                or exists(join(path, split, "state.json"))
            ):
                return False
        return True
    if required_splits == {"train"}:
        return exists(join(path, "dataset_info.json")) or exists(join(path, "state.json"))
    return False


def _ensure_processed_dataset(path: str, download_fn, required_splits=None):
    lock_path = f"{path}.lock"
    while True:
        if exists(path) and _has_required_splits(path, required_splits):
            return
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
        except FileExistsError:
            acquired = False
        if not acquired:
            time.sleep(2)
            continue
        try:
            if not exists(path) or not _has_required_splits(path, required_splits):
                logging.info(
                    "Processed dataset missing or invalid at %s; building from local source.", path
                )
                if exists(path):
                    shutil.rmtree(path)
                download_fn(_pixmo_cache_only())
        finally:
            if exists(lock_path):
                os.unlink(lock_path)
        return



class PixMoCount(Dataset):
    
    def __init__(self, split, sample=None, counting=False, 
                 csv_dir="/Pixmo_Dataset/pixmo-count",
                 images_dir="/Pixmo_Dataset/pixmo-count/images",
                 keep_in_memory=False):
        """
        Args:
            split: 'train', 'validation', or 'test'
            sample: optional, limit dataset to first N samples
            counting: if True, style is 'point_count', else 'pointing'
        """
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"Unknown split {split}")
        
        self.counting = counting
        self._split = split
        self.images_dir = images_dir
        
        # Load CSV files
        print(f"Loading CSV for {split} split")
        
        if split == "train":
            # Combine all splits for training
            df = pd.read_csv(os.path.join(csv_dir, "train_combined_verified_strict.csv"))
        else:
            # Use specific split
            csv_files = {
                "validation": "validation_clean.csv",
                "test": "test_clean.csv"
            }
            df = pd.read_csv(os.path.join(csv_dir, csv_files[split]))
        
        # Apply sampling if requested
        if sample is not None and sample > 0 and len(df) > sample:
            df = df.iloc[:sample].reset_index(drop=True)
            print(f"Sampled to {len(df)} examples")
        
        self.df = df
        print(f"Loaded {len(self.df)} examples for {split} split")
        print("Note: Images and points will be loaded lazily during data loading")

    def _parse_points(self, points_str):
        """
        Parse points from CSV string format
        Returns dict with 'x' and 'y' arrays
        """
        try:
            if pd.isna(points_str) or not isinstance(points_str, str):
                return {'x': [], 'y': []}
            
            # Use regex to extract all {'x': value, 'y': value} patterns
            pattern = r"\{'x':\s*([\d.]+),\s*'y':\s*([\d.]+)\}"
            matches = re.findall(pattern, points_str)
            
            if len(matches) == 0:
                return {'x': [], 'y': []}
            
            # Separate x and y coordinates
            x_coords = [float(x) for x, y in matches]
            y_coords = [float(y) for x, y in matches]
            
            return {'x': x_coords, 'y': y_coords}
        except Exception as e:
            print(f"Error parsing points: {e}")
            return {'x': [], 'y': []}

    def __len__(self):
        return len(self.df)

    def get(self, item, rng=None):
        """Get item - lazily loads image and processes data on-the-fly"""
        
        row = self.df.iloc[item]
        
        # Load image
        saved_filename = row['saved_filename']
        image_path = os.path.join(self.images_dir, saved_filename)
        
        try:
            if isinstance(image_path, str) and os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")
        except Exception as e:
            raise FileNotFoundError(f"Could not load image: {image_path}. Error: {e}")
        
        # Build output
        out = dict(
            style="point_count" if self.counting else "pointing",
            image=image,
            label=_safe_text(row['label']),
            metadata=dict(
                image_url=row['image_url'],
                count=int(row['count']) if pd.notna(row['count']) else 0,
            )
        )
        
        # Add points for train split
        if self._split == "train" and 'points' in row and pd.notna(row['points']):
            points_dict = self._parse_points(row['points'])
            if len(points_dict['x']) > 0:
                out["points"] = np.stack([points_dict["x"], points_dict["y"]], -1, dtype=np.float32)
            else:
                out["points"] = np.array([], dtype=np.float32).reshape(0, 2)
        
        return out




class PixMoDocs(Dataset):
    V1_STYLE = {
        "pixmo_docs_other": "scifi_document",
        "pixmo_docs_charts": "scifi_charts",
        "pixmo_docs_diagrams": "scifi_diagram",
        "pixmo_docs_tables": "scifi_table"
    }

    @classmethod
    def download(cls, n_procs=1):
        for name in ["other", "charts", "diagrams", "tables"]:
            local_path = _local_dataset_path("pixmo-docs")
            datasets.load_dataset_builder(local_path, name=name).download_and_prepare()

    def __init__(self, doc_type, split, sample=None, keep_in_memory=False, v1_style=False):
        assert doc_type in ["other", "charts", "diagrams", "tables"]
        assert split in ["train", "validation", "test"]
        self.doc_type = doc_type
        self.v1_style = v1_style
        self._split = split
        docs_root = _local_dataset_path("pixmo-docs")
        doc_dir = join(docs_root, doc_type)
        
        # Handle test split
        if split == "test" and not _split_exists(doc_dir, "test"):
            if _split_exists(doc_dir, "validation"):
                logging.warning(
                    "Test split not found for pixmo-docs/%s; using validation split.",
                    doc_type,
                )
                split = "validation"
            else:
                logging.warning(
                    "Test/validation splits not found for pixmo-docs/%s; sampling 500 from train.",
                    doc_type,
                )
                # FIXED: Load from train split, not validation
                train_ds = datasets.load_dataset(
                    docs_root,
                    name=doc_type,
                    split="train",
                    keep_in_memory=keep_in_memory
                )
                # Use a different seed for test to avoid overlap with validation
                self.dataset = _sample_hf_dataset(train_ds, 500, seed=1)
                debug_n = _debug_sample_size()
                if debug_n is not None:
                    self.dataset = _sample_hf_dataset(self.dataset, debug_n, seed=1)
                return
        
        # Handle validation split
        elif split == "validation" and not _split_exists(doc_dir, "validation"):
            logging.warning(
                "Validation split not found for pixmo-docs/%s; sampling 500 from train.",
                doc_type,
            )
            # FIXED: Load from train split (this was correct)
            train_ds = datasets.load_dataset(
                docs_root,
                name=doc_type,
                split="train",
                keep_in_memory=keep_in_memory
            )
            # Use seed=0 for validation sampling
            self.dataset = _sample_hf_dataset(train_ds, 500, seed=0)
            debug_n = _debug_sample_size()
            if debug_n is not None:
                self.dataset = _sample_hf_dataset(self.dataset, debug_n, seed=0)
            return
        
        # Load the actual split if it exists
        self.dataset = datasets.load_dataset(
            docs_root,
            name=doc_type,
            split=split,
            keep_in_memory=keep_in_memory
        )
        debug_n = _debug_sample_size()
        if debug_n is not None and split != "train":
            self.dataset = _sample_hf_dataset(self.dataset, debug_n, seed=0)

    def __len__(self):
        return len(self.dataset)

    def get(self, item, rng):
        style = f"pixmo_docs_{self.doc_type}"
        if self.v1_style:
            style = self.V1_STYLE[style]
        example = self.dataset[item]
        qas = example["questions"]
        questions = qas["question"]
        answers = qas["answer"]
        if self._split == "train":
            return dict(
                image=example["image"],
                style=style,
                message_list=[
                    dict(question=_safe_text(q), answer=_safe_text(a), style=style) for q, a in
                    zip(questions, answers)
                ],
                metadata=dict(
                    image_id=example["image_id"]
                )
            )
        if isinstance(questions, list) and len(questions) > 0:
            idx = int(rng.randint(len(questions))) if rng is not None else 0
            question = _safe_text(questions[idx])
            if isinstance(answers, list) and len(answers) > idx:
                answer = _safe_text(answers[idx])
            else:
                answer = _safe_text(answers)
        else:
            question = _safe_text(questions)
            answer = _safe_text(answers)
        return dict(
            image=example["image"],
            question=question,
            answers=answer,
            style=style,
            metadata=dict(
                image_id=example["image_id"]
            )
        )




class PixMoPoints(Dataset):

    def __init__(self, split, kind="both", counting=False, 
                 csv_path="/Pixmo_Dataset/pixmo-points/pixmo_points_clean_verified_strict_1.csv",
                 images_dir="/Pixmo_Dataset/pixmo-points/images_1",
                 keep_in_memory=False):
        """
        Args:
            split: 'train' or 'validation'
            kind: 'high_frequency', 'basic', or 'both'
            counting: if False, mode is 'point_count', if True, mode is 'pointing'
        """
        if kind not in ["high_frequency", "basic", "both"]:
            raise ValueError(kind)
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        
        mode = "pointing" if counting else "point_count"
        self.split = split
        self.kind = kind
        self.mode = mode
        self.images_dir = images_dir
        
        # Load CSV (fast)
        print(f"Loading CSV from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Filter by collection_method based on 'kind' (fast)
        if kind == "basic":
            df = df[df['collection_method'] == 'pointing'].reset_index(drop=True)
        elif kind == "high_frequency":
            df = df[df['collection_method'] == 'counting'].reset_index(drop=True)
        
        # Get unique image URLs
        unique_image_urls = df['image_url'].unique().tolist()
        
        # Simple split: train = all, validation = first 200
        if split == "train":
            self.image_urls = unique_image_urls[200: ]
        else:  # validation
            self.image_urls = unique_image_urls[:200]
        
        # Store the full dataframe
        self.df = df
        
        print(f"Loaded {len(self.image_urls)} images for {split} split (kind={kind}, mode={self.mode})")
        print("Note: Points parsing will happen lazily during data loading")

    def _parse_points(self, points_str):
        """
        Parse points from CSV string format to list of dicts with x, y coordinates
        This is called lazily in get() method
        """
        try:
            if not isinstance(points_str, str):
                return []
            
            # Use regex to extract all {'x': value, 'y': value} patterns
            pattern = r"\{'x':\s*([\d.]+),\s*'y':\s*([\d.]+)\}"
            matches = re.findall(pattern, points_str)
            
            # Convert to list of dicts
            result = [{'x': float(x), 'y': float(y)} for x, y in matches]
            
            return result
        except Exception as e:
            print(f"Error parsing points: {e}")
            return []

    def __len__(self):
        return len(self.image_urls)

    def get(self, item, rng=None):
        """Get item - lazily processes data on-the-fly"""
        
        # Get the image URL for this item
        image_url = self.image_urls[item]
        
        # Filter dataframe for this specific image (fast lookup)
        image_rows = self.df[self.df['image_url'] == image_url]
        
        # Get saved filename (same for all rows of this image)
        saved_filename = image_rows['saved_filename'].iloc[0]
        
        # Load image
        image_path = os.path.join(self.images_dir, saved_filename)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Could not load image: {image_path}. Error: {e}")
        
        # Parse points and build messages (done lazily here)
        messages = []
        for _, row in image_rows.iterrows():
            label = _safe_text(row['label'])
            
            # Parse points on-the-fly
            if pd.notna(row['points']):
                points = self._parse_points(row['points'])
            else:
                points = []
            
            # Build message in original format
            if len(points) > 0:
                points_array = np.stack(
                    [[p['x'] for p in points], [p['y'] for p in points]], 
                    axis=-1
                )
            else:
                points_array = np.array([]).reshape(0, 2)
            
            messages.append(dict(
                label=label,
                points=points_array,
                point_scale=100,
                style=self.mode
            ))
        
        return dict(
            image=image,
            message_list=messages,
            metadata=dict(
                image_url=image_url,
            )
        )


class PixMoPointExplanations(Dataset):

    def __init__(self, split, split_groups=True,
                 csv_path="/Pixmo_Dataset/pixmo-points-explanations/train_verified_strict.csv",
                 images_dir="/Pixmo_Dataset/pixmo-points-explanations/images",
                 keep_in_memory=False):
        """
        Args:
            split: 'train' or 'validation'
            split_groups: if True, splits message lists > 1 in half
        """
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        
        self.split = split
        self.split_groups = split_groups
        self.images_dir = images_dir
        
        # Load CSV
        print(f"Loading CSV from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Filter out rows where parsed_response is null
        df = df[df['parsed_response'].notna()].reset_index(drop=True)
        
        # Get unique image URLs
        unique_image_urls = df['image_url'].unique().tolist()
        
        # Simple split: train = all, validation = first 200
        if split == "train":
            self.image_urls = unique_image_urls[200:]
        else:  # validation
            self.image_urls = unique_image_urls[:200]
        
        # Store the full dataframe for lookup
        self.df = df
        
        print(f"Loaded {len(self.image_urls)} images for {split} split")
        print("Note: Parsing will happen lazily during data loading")

    def _parse_list(self, list_str):
        """Parse a list from string format or return plain string wrapped in list"""
        try:
            if pd.isna(list_str):
                return []
            if isinstance(list_str, list):
                return list_str
            if isinstance(list_str, str):
                # Check if it looks like a list representation
                list_str_stripped = list_str.strip()
                if list_str_stripped.startswith('[') and list_str_stripped.endswith(']'):
                    try:
                        return ast.literal_eval(list_str_stripped)
                    except:
                        # If parsing fails, return as single-item list
                        return [list_str]
                else:
                    # It's just a plain string, return as single-item list
                    return [list_str]
            return []
        except Exception as e:
            print(f"Error parsing list: {e}, input: {str(list_str)[:100]}")
            return [list_str] if isinstance(list_str, str) else []

    def _parse_points(self, points_str):
        """
        Parse points from string format to list of [x, y] coordinates
        """
        try:
            if pd.isna(points_str):
                return []
            
            if isinstance(points_str, str):
                # Use regex to extract all {'x': value, 'y': value} patterns
                pattern = r"\{'x':\s*([\d.]+),\s*'y':\s*([\d.]+)\}"
                matches = re.findall(pattern, points_str)
                if matches:
                    return [[float(x), float(y)] for x, y in matches]
                
                # Try to parse as array string like "array([70.4,  4.9])"
                array_pattern = r"array\(\[([\d.,\s]+)\]"
                array_match = re.search(array_pattern, points_str)
                if array_match:
                    coords = array_match.group(1).split(',')
                    if len(coords) == 2:
                        return [[float(coords[0].strip()), float(coords[1].strip())]]
                
                return []
            elif isinstance(points_str, (list, np.ndarray)):
                # Already parsed
                return self._normalize_points(points_str)
            return []
        except Exception as e:
            print(f"Error parsing points: {e}, input: {str(points_str)[:100]}")
            return []

    def _normalize_points(self, points):
        """Normalize points to [[x, y], [x, y], ...] format"""
        if not points:
            return []
        
        # Handle numpy arrays
        if isinstance(points, np.ndarray):
            if points.ndim == 1 and len(points) == 2:
                return [[float(points[0]), float(points[1])]]
            elif points.ndim == 2:
                return [[float(p[0]), float(p[1])] for p in points]
            return []
        
        # Handle dict format
        if isinstance(points, dict):
            if "x" in points and "y" in points:
                return [[float(points["x"]), float(points["y"])]]
            return []
        
        # Handle list/tuple
        if isinstance(points, (list, tuple)):
            if len(points) == 0:
                return []
            
            first = points[0]
            
            # List of dicts
            if isinstance(first, dict):
                out = []
                for p in points:
                    if isinstance(p, dict) and "x" in p and "y" in p:
                        out.append([float(p["x"]), float(p["y"])])
                return out
            
            # List of lists/tuples (already in correct format)
            if isinstance(first, (list, tuple, np.ndarray)):
                if len(first) == 2:
                    return [[float(p[0]), float(p[1])] for p in points]
            
            # Single [x, y] point
            if len(points) == 2 and isinstance(points[0], (int, float, np.number)):
                return [[float(points[0]), float(points[1])]]
        
        return []

    def _parse_nested_list(self, nested_str):
        """
        Parse nested list structures like [[...], [...]]
        """
        try:
            if pd.isna(nested_str):
                return []
            if isinstance(nested_str, (list, np.ndarray)):
                return list(nested_str)
            if isinstance(nested_str, str):
                # Try ast.literal_eval first
                try:
                    return ast.literal_eval(nested_str)
                except:
                    # Fallback: return empty list
                    return []
            return []
        except Exception as e:
            print(f"Error parsing nested list: {e}, input: {str(nested_str)[:100]}")
            return []

    def __len__(self):
        return len(self.image_urls)

    def get(self, item, rng):
        """Get item - lazily loads and processes data"""
        
        # Get the image URL for this item
        image_url = self.image_urls[item]
        
        # Filter dataframe for this specific image
        image_rows = self.df[self.df['image_url'] == image_url]
        
        # Take the first row for image info
        first_row = image_rows.iloc[0]
        saved_filename = first_row['saved_filename']
        
        # Load image
        image_path = join(self.images_dir, saved_filename)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Could not load image: {image_path}. Error: {e}")
        
        # Parse all fields for this image (combine all rows if multiple)
        all_questions = []
        all_responses = []
        all_alt_texts = []
        all_inline_texts = []
        all_points = []
        
        for _, row in image_rows.iterrows():
            # These are simple strings, not lists
            questions = self._parse_list(row['question'])
            responses = self._parse_list(row['parsed_response'])
            alt_texts = self._parse_nested_list(row['alt_text']) if 'alt_text' in row else []
            inline_texts = self._parse_nested_list(row['inline_text']) if 'inline_text' in row else []
            points = self._parse_nested_list(row['points']) if 'points' in row else []
            
            # Extend lists
            all_questions.extend(questions if isinstance(questions, list) else [questions])
            all_responses.extend(responses if isinstance(responses, list) else [responses])
            all_alt_texts.extend(alt_texts if isinstance(alt_texts, list) else [alt_texts])
            all_inline_texts.extend(inline_texts if isinstance(inline_texts, list) else [inline_texts])
            all_points.extend(points if isinstance(points, list) else [points])
        
        # Build message list
        msg_list = []
        max_len = max(len(all_questions), len(all_responses), len(all_alt_texts), 
                      len(all_inline_texts), len(all_points))
        
        for i in range(max_len):
            q = _safe_text(all_questions[i]) if i < len(all_questions) else ""
            res = _safe_text(all_responses[i]) if i < len(all_responses) else ""
            alt = all_alt_texts[i] if i < len(all_alt_texts) else []
            inline = all_inline_texts[i] if i < len(all_inline_texts) else []
            points = all_points[i] if i < len(all_points) else []
            
            # Ensure alt, inline, points are lists
            if not isinstance(alt, list):
                alt = [alt] if alt else []
            if not isinstance(inline, list):
                inline = [inline] if inline else []
            if not isinstance(points, (list, np.ndarray)):
                points = [points] if points else []
            
            # Parse points for each annotation
            parsed_points = []
            for p in points:
                if isinstance(p, str):
                    parsed_points.append(self._normalize_points(self._parse_points(p)))
                elif isinstance(p, (list, np.ndarray)):
                    parsed_points.append(self._normalize_points(p))
                else:
                    parsed_points.append([])
            
            # Build annotations
            max_annotations = max(len(parsed_points), len(inline), len(alt))
            annotations = []
            for j in range(max_annotations):
                p = parsed_points[j] if j < len(parsed_points) else []
                i_text = _safe_text(inline[j]) if j < len(inline) else ""
                a_text = _safe_text(alt[j]) if j < len(alt) else ""
                annotations.append(dict(
                    points=p,
                    inline_text=i_text,
                    alt_text=a_text
                ))

            # Handle <|POINT|> placeholders
            if isinstance(res, str):
                placeholder_count = res.count("<|POINT|>")
            else:
                placeholder_count = 0
            
            if placeholder_count == 0:
                annotations = []
            elif placeholder_count < len(annotations):
                annotations = annotations[:placeholder_count]
            
            if placeholder_count > 0 and not annotations and isinstance(res, str):
                res = res.replace("<|POINT|>", "").strip()
            
            msg_list.append(dict(
                question=q,
                answer=res,
                answer_annotations=annotations,
                style="point_qa"
            ))
        
        # Create base example
        molmo_ex = dict(
            image=image,
            metadata=dict(
                image_url=image_url,
            )
        )
        
        # Apply split_groups logic
        if self.split_groups and len(msg_list) > 1:
            n = len(msg_list) // 2 + len(msg_list) % 2
            return dict(molmo_ex, message_list=msg_list[:n])
        else:
            return dict(molmo_ex, message_list=msg_list)






class PixMoCapQa(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=False, n_val=2048, cache_only=False):
        """Not needed - using local CSV and images"""
        local_name = join(PIXMO_DATASETS, "pixmo-cap-qa")
        if not exists(local_name):
            raise FileNotFoundError(
                f"Dataset directory not found: {local_name}\n"
                f"Expected CSV file: {join(local_name, 'pixmo_cap_qa_clean_verified_strict_1.csv')}\n"
                f"Expected images in: {join(local_name, 'images')}"
            )
        return

    def __init__(self, split="train", prefix_how_many=True, keep_in_memory=False, n_val=2048):
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}. Must be 'train' or 'validation'")
        
        self.split = split
        self.prefix_how_many = prefix_how_many
        self.dataset_root = join(PIXMO_DATASETS, "pixmo-cap-qa")
        self.image_dir = join(self.dataset_root, "images")
        
        # Load CSV
        csv_path = join(self.dataset_root, "pixmo_cap_qa_clean_verified_strict_1.csv")
        
        # Check if files exist
        if not exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        print(f"Loading data from {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} total samples")
        
        # Add full image path
        df['image_path'] = df['saved_filename'].apply(
            lambda x: join(self.image_dir, x)
        )
        
        # Group by image_url to get all QA pairs for each image
        print("Grouping by image_url...")
        grouped = df.groupby('image_url').agg({
            'image_path': 'first',  # Same image for all rows with same URL
            'saved_filename': 'first',
            'question': list,  # Collect all questions
            'answer': list,    # Collect all answers
        }).reset_index()
        
        print(f"Grouped into {len(grouped)} unique images")
        
        # Create train/validation split
        if split == "validation":
            self.data = grouped.tail(n_val).reset_index(drop=True)
        else:  # train
            self.data = grouped.head(len(grouped) - n_val).reset_index(drop=True)
        
        print(f"Split '{split}': {len(self.data)} samples")
        
        # Verify required columns
        required_cols = ['image_url', 'image_path', 'saved_filename', 'question', 'answer']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check if images exist (sample check)
        sample_size = min(10, len(self.data))
        missing_images = []
        for idx in range(sample_size):
            img_path = self.data.iloc[idx]['image_path']
            if not exists(img_path):
                missing_images.append(img_path)
        
        if missing_images:
            print(f"Warning: {len(missing_images)}/{sample_size} sample images not found")
            print(f"Example missing: {missing_images[0]}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        """Standard PyTorch dataset interface"""
        rng = np.random.RandomState()
        return self.get(item, rng)

    def get(self, item, rng):
        """Get item with optional random number generator"""
        row = self.data.iloc[item]
        
        # Load image
        image_path = row['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
        
        # Get questions and answers
        questions = row['question']
        answers = row['answer']
        
        # Handle case where questions/answers might be strings (single Q/A)
        if isinstance(questions, str):
            questions = [questions]
        if isinstance(answers, str):
            answers = [answers]
        
        # Build message list (alternating user questions and assistant answers)
        messages_for_image = []
        for q, a in zip(questions, answers):
            conversation = [_safe_text(q), _safe_text(a)]  # [user_question, assistant_answer]
            messages_for_image.append({
                "messages": conversation,
                "style": "synthetic_qa"
            })
        
        ex = dict(
            image=image,
            message_list=messages_for_image,
            metadata=dict(
                image_url=row['image_url'],
                saved_filename=row['saved_filename'],
            )
        )
        
        # Add prefix to "how many" questions if enabled
        if self.prefix_how_many:
            for conv in ex["message_list"]:
                messages = conv["messages"]
                # Check user questions (at even indices: 0, 2, 4, ...)
                for user_question_ix in range(0, len(messages), 2):
                    if re.fullmatch("how many.*", messages[user_question_ix].lower()):
                        prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                        messages[user_question_ix] = prefix + messages[user_question_ix]
        
        return ex




class PixMoCap(Dataset):
    
    def __init__(self, split, mode, prefix_how_many=True,
                 csv_path="/Pixmo_Dataset/pixmo-cap/pixmo_cap_clean_verified_strict_1.csv",
                 images_dir="/Pixmo_Dataset/pixmo-cap/images_1",
                 keep_in_memory=False):
        """
        Args:
            split: 'train' or 'validation'
            mode: 'transcripts', 'captions', 'transcript_and_caption', or 'transcript1_and_caption'
            prefix_how_many: for compatibility (not used in this dataset)
        """
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        if mode not in ["transcripts", "captions", "transcript_and_caption", "transcript1_and_caption"]:
            raise ValueError(mode)
        
        self.split = split
        self.mode = mode
        self.images_dir = images_dir
        
        # Load CSV
        print(f"Loading CSV from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Get unique image URLs
        unique_image_urls = df['image_url'].unique().tolist()
        
        # Simple split: train = all, validation = first 200
        if split == "train":
            self.image_urls = unique_image_urls[200:]
        else:  # validation
            self.image_urls = unique_image_urls[:200]
        
        # Store the full dataframe for lookup
        self.df = df
        
        print(f"Loaded {len(self.image_urls)} images for {split} split (mode={mode})")
        print("Note: Captions and transcripts will be loaded lazily")

    def _parse_transcripts(self, transcripts_str):
        """
        Parse transcripts from string format (should be a list)
        """
        try:
            if pd.isna(transcripts_str):
                return []
            if isinstance(transcripts_str, list):
                return transcripts_str
            if isinstance(transcripts_str, str):
                # Try to parse as list
                try:
                    parsed = ast.literal_eval(transcripts_str)
                    if isinstance(parsed, list):
                        return parsed
                    else:
                        return [str(parsed)]
                except:
                    # If it's just a single string, wrap it in a list
                    return [transcripts_str]
            return []
        except Exception as e:
            print(f"Error parsing transcripts: {e}")
            return []

    def __len__(self):
        return len(self.image_urls)

    def get(self, item, rng):
        """Get item - lazily loads image and processes captions/transcripts"""
        
        # Get the image URL for this item
        image_url = self.image_urls[item]
        
        # Filter dataframe for this specific image (should be one row typically)
        image_rows = self.df[self.df['image_url'] == image_url]
        
        # Take the first row
        row = image_rows.iloc[0]
        saved_filename = row['saved_filename']
        
        # Load image
        image_path = join(self.images_dir, saved_filename)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Could not load image: {image_path}. Error: {e}")
        
        # Get caption and transcripts
        caption = _safe_text(row['caption']) if 'caption' in row and pd.notna(row['caption']) else ""
        transcripts = [_safe_text(tr) for tr in self._parse_transcripts(row['transcripts'])] if 'transcripts' in row else []
        
        # Build message list based on mode
        messages = []
        
        if self.mode in ["captions", "transcript_and_caption", "transcript1_and_caption"]:
            messages.append(dict(text=caption, style="long_caption"))
        
        if self.mode in ["transcript_and_caption", "transcript1_and_caption"]:
            if len(transcripts) > 0:
                if self.mode == "transcript_and_caption":
                    # Randomly select a transcript
                    if rng is not None:
                        ix = rng.randint(0, len(transcripts))
                    else:
                        ix = 0
                else:  # transcript1_and_caption
                    ix = 0
                messages.append(dict(text=transcripts[ix], style="transcript"))
        
        if self.mode == "transcripts":
            messages += [dict(text=tr, style="transcript") for tr in transcripts]
        
        out = dict(
            image=image,
            message_list=messages,
            metadata=dict(
                image_url=image_url,
            )
        )
        
        return out







class PixMoAskModelAnything(Dataset):
    
    def __init__(self, split, prefix_how_many=True,
                 csv_path="/Pixmo_Dataset/pixmo-ask-model-anything/image_mapping_ask_any_verified_strict_1.csv",
                 images_dir="/Pixmo_Dataset/pixmo-ask-model-anything/images/ask-anything",
                 keep_in_memory=False):
        """
        Args:
            split: 'train' or 'validation'
            prefix_how_many: if True, adds prefix to "how many" questions
        """
        if split not in ["train", "validation"]:
            raise ValueError(f"Unknown split {split}")
        
        self.split = split
        self.prefix_how_many = prefix_how_many
        self.images_dir = images_dir
        
        # Load CSV
        print(f"Loading CSV from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Get unique image URLs
        unique_image_urls = df['image_url'].unique().tolist()
        
        # Simple split: train = all, validation = first 200
        if split == "train":
            self.image_urls = unique_image_urls[200:]
        else:  # validation
            self.image_urls = unique_image_urls[:200]
        
        # Store the full dataframe for lookup
        self.df = df
        
        print(f"Loaded {len(self.image_urls)} images for {split} split")
        print("Note: Questions/answers will be grouped and loaded lazily")

    def _parse_qa_list(self, qa_str):
        """
        Parse question or answer list from string format
        Handles formats like: "['question1', 'question2']" or similar
        """
        try:
            if pd.isna(qa_str):
                return []
            
            if isinstance(qa_str, str):
                import ast
                try:
                    # Try ast.literal_eval for clean list strings
                    parsed = ast.literal_eval(qa_str)
                    if isinstance(parsed, list):
                        return parsed
                    return [parsed]
                except:
                    # Fallback: extract strings between quotes
                    pattern = r"['\"]([^'\"]*)['\"]"
                    matches = re.findall(pattern, qa_str)
                    return matches if matches else []
            elif isinstance(qa_str, list):
                return qa_str
            else:
                return []
        except Exception as e:
            print(f"Error parsing Q/A list: {e}")
            return []

    def __len__(self):
        return len(self.image_urls)

    def get(self, item, rng):
        """Get item - lazily loads image and processes Q&A pairs"""
        
        # Get the image URL for this item
        image_url = self.image_urls[item]
        
        # Filter dataframe for this specific image
        image_rows = self.df[self.df['image_url'] == image_url]
        
        # Get saved filename (should be same for all rows)
        saved_filename = image_rows['saved_filename'].iloc[0]
        
        # Load image
        image_path = join(self.images_dir, saved_filename)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise FileNotFoundError(f"Could not load image: {image_path}. Error: {e}")
        
        # Collect all questions and answers for this image
        messages = []
        for _, row in image_rows.iterrows():
            # Parse questions and answers
            questions = self._parse_qa_list(row['question']) if 'question' in row else []
            answers = self._parse_qa_list(row['answer']) if 'answer' in row else []
            
            # Handle single question/answer (not in list format)
            if not questions and 'question' in row and pd.notna(row['question']):
                if not isinstance(row['question'], str) or row['question'].startswith('['):
                    questions = self._parse_qa_list(row['question'])
                else:
                    questions = [row['question']]
            
            if not answers and 'answer' in row and pd.notna(row['answer']):
                if not isinstance(row['answer'], str) or row['answer'].startswith('['):
                    answers = self._parse_qa_list(row['answer'])
                else:
                    answers = [row['answer']]

            if not isinstance(questions, list):
                questions = [questions]
            if not isinstance(answers, list):
                answers = [answers]

            # Pair up questions and answers
            for q, a in zip(questions, answers):
                messages.append(dict(
                    question=_safe_text(q),
                    answer=_safe_text(a),
                    style="user_qa"
                ))
        
        ex = dict(
            image=image,
            message_list=messages,
            metadata=dict(
                image_url=image_url,
            )
        )
        
        # Apply "how many" prefix if enabled
        if self.prefix_how_many and rng is not None:
            for conv in ex["message_list"]:
                if re.fullmatch(r"how many.*", conv["question"].lower()):
                    prefix = NO_POINT_PREFIX[rng.randint(0, len(NO_POINT_PREFIX))]
                    conv["question"] = prefix + conv["question"]
        
        return ex


class PixMoPointsEval(Dataset):
    @classmethod
    def download(cls, n_procs=1, check_sha=True, cache_only=False):
        """Not needed - using local CSV and images"""
        local_name = join(PIXMO_DATASETS, "pixmo-points-eval")
        if not exists(local_name):
            raise FileNotFoundError(
                f"Dataset directory not found: {local_name}\n"
                f"Expected CSV files:\n"
                f"  - {join(local_name, 'train_combined_verified_strict.csv')}\n"
                f"  - {join(local_name, 'test_clean_verified_strict.csv')}\n"
                f"Expected images in: {join(local_name, 'images')}"
            )
        return

    def __init__(self, split="test", keep_in_memory=False):
        if split not in ["train", "test"]:
            raise ValueError(f"Unknown split {split}. Must be 'train' or 'test'")
        
        # Set paths
        self.split = split
        self.dataset_root = join(PIXMO_DATASETS, "pixmo-points-eval")
        self.image_dir = join(self.dataset_root, "images")
        
        # Load appropriate CSV
        if split == "train":
            csv_path = join(self.dataset_root, "train_combined_verified_strict.csv")
        else:  # test
            csv_path = join(self.dataset_root, "test_clean_verified_strict.csv")
        
        # Check if files exist
        if not exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not exists(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Load CSV
        print(f"Loading {split} data from {csv_path}")
        self.data = pd.read_csv(csv_path)
        print(f"Loaded {len(self.data)} samples for {split} split")
        
        # Verify required columns
        required_cols = ['saved_filename', 'label', 'points', 'masks', 'image_url']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def _parse_points(self, points_str):
        """Parse points from string format"""
        try:
            if pd.isna(points_str):
                return []
            
            if isinstance(points_str, str):
                # Use regex to extract all {'x': value, 'y': value} patterns
                pattern = r"\{'x':\s*([\d.]+),\s*'y':\s*([\d.]+)\}"
                matches = re.findall(pattern, points_str)
                return [{'x': float(x), 'y': float(y)} for x, y in matches]
            elif isinstance(points_str, list):
                return points_str
            else:
                return []
        except Exception as e:
            print(f"Error parsing points: {e}")
            return []

    def _parse_masks(self, masks_str):
        """Parse masks from string format - more robust parsing"""
        try:
            if pd.isna(masks_str):
                return np.array([], dtype=bool)
            
            if isinstance(masks_str, str):
                # Remove common prefixes that cause issues
                masks_str = masks_str.strip()
                
                # Remove numpy array calls if present
                if masks_str.startswith('np.array(') or masks_str.startswith('numpy.array('):
                    # Extract content between parentheses
                    match = re.search(r'(?:np|numpy)\.array\((.*)\)', masks_str, re.DOTALL)
                    if match:
                        masks_str = match.group(1)
                
                # Remove dtype specifications
                masks_str = re.sub(r',\s*dtype\s*=\s*[^,\)]+', '', masks_str)
                
                # Try to find boolean values or 0/1 patterns
                # Match patterns like [True, False, True] or [1, 0, 1] or [ True  False  True]
                
                # First try: space-separated boolean/int values inside brackets
                if '[' in masks_str and ']' in masks_str:
                    # Extract content between outermost brackets
                    inner = re.search(r'\[(.*)\]', masks_str, re.DOTALL)
                    if inner:
                        content = inner.group(1)
                        # Replace True/False with 1/0
                        content = content.replace('True', '1').replace('False', '0')
                        # Split by whitespace or comma and filter out empty strings
                        values = [v.strip() for v in re.split(r'[,\s]+', content) if v.strip()]
                        # Convert to boolean
                        return np.array([bool(int(v)) for v in values if v in ['0', '1']], dtype=bool)
                
                # Second try: use ast.literal_eval for clean formats
                try:
                    import ast
                    parsed = ast.literal_eval(masks_str)
                    return np.array(parsed, dtype=bool)
                except:
                    pass
                
                # Third try: extract all True/False or 0/1 values
                true_false = re.findall(r'\b(True|False)\b', masks_str)
                if true_false:
                    return np.array([v == 'True' for v in true_false], dtype=bool)
                
                ones_zeros = re.findall(r'\b([01])\b', masks_str)
                if ones_zeros:
                    return np.array([bool(int(v)) for v in ones_zeros], dtype=bool)
                
                # If nothing worked, return empty array
                print(f"Warning: Could not parse masks, returning empty array. Sample: {masks_str[:100]}")
                return np.array([], dtype=bool)
            
            elif isinstance(masks_str, (list, np.ndarray)):
                return np.array(masks_str, dtype=bool)
            else:
                return np.array([], dtype=bool)
                
        except Exception as e:
            print(f"Error parsing masks: {e}, masks_str sample: {str(masks_str)[:100]}")
            return np.array([], dtype=bool)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        """Standard PyTorch dataset interface"""
        return self.get(item, None)

    def get(self, item, rng):
        """Get item with optional random number generator"""
        row = self.data.iloc[item]
        
        # Load image
        image_path = join(self.image_dir, row['saved_filename'])
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
        
        # Parse points
        points_data = self._parse_points(row['points'])
        
        # Convert points to numpy array [N, 2]
        if isinstance(points_data, list) and len(points_data) > 0:
            points = np.stack(
                [[p["x"] for p in points_data], [p["y"] for p in points_data]], 
                axis=-1
            )
        else:
            points = np.array([]).reshape(0, 2)
        
        # Parse masks with robust parsing
        masks = self._parse_masks(row['masks'])
        
        return dict(
            image=image,
            label=_safe_text(row['label']),
            points=points,
            point_scale=100,
            style="pointing",
            metadata=dict(
                points=points,
                masks=masks,
                image_url=row['image_url'],
                saved_filename=row['saved_filename'],
            )
    )
