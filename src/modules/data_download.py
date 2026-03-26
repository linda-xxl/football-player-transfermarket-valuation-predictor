"""
Data download module.

Downloads the Transfermarkt dataset from Kaggle using kagglehub and copies
all CSV files into the project data directory.

Usage:
    from src.modules.data_download import download
    download()
"""

import logging
import shutil
from pathlib import Path

import kagglehub

from .config import DATA_DIR, KAGGLE_DATASET

logger = logging.getLogger(__name__)


def download(target_dir: Path = DATA_DIR, dataset: str = KAGGLE_DATASET) -> Path:
    """
    Download the Kaggle dataset and copy files to ``target_dir``.

    Parameters
    ----------
    target_dir : Path
        Destination directory for raw CSV files.
    dataset : str
        Kaggle dataset slug (e.g. ``"davidcariboo/player-scores"``).

    Returns
    -------
    Path
        The target directory containing the downloaded files.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset '%s' from Kaggle …", dataset)
    cache_path = kagglehub.dataset_download(dataset)
    cache_path = Path(cache_path)
    logger.info("Kaggle cache path: %s", cache_path)

    # Copy every file from the cache into the project data directory
    copied = 0
    for item in cache_path.iterdir():
        dest = target_dir / item.name
        if item.is_file():
            shutil.copy2(item, dest)
            copied += 1
        elif item.is_dir():
            shutil.copytree(item, dest, dirs_exist_ok=True)
            copied += 1

    logger.info("Copied %d item(s) to %s", copied, target_dir)
    return target_dir
