import time
from functools import wraps
import requests
from tqdm import tqdm
import os


def numerize(n: int) -> str:
    """display positive number n for readability

    Examples:
        >>> numerize(1_000)
        '1K'
        >>> numerize(1_000_000_000)
        '1B'
    """
    sufix2upbound = {
        "EMPTY": 1e3,
        "K": 1e6,
        "M": 1e9,
        "B": 1e12,
        "END": float("inf"),
    }

    display_n, sufix = n, ""
    for s, base in sufix2upbound.items():
        # number >= 1000B will alway have sufix 'B'
        if s == "END":
            display_n = int(n / 1e9)
            sufix = "B"
            break

        if n < base:
            sufix = "" if s == "EMPTY" else s
            display_n = int(n / (base / 1e3))
            break
    return f"{display_n}{sufix}"


def time_it(func: any):
    """returns result and elapsed time"""

    @wraps(func)
    def inner(*args, **kwargs):
        pref = time.perf_counter()
        result = func(*args, **kwargs)
        delta = time.perf_counter() - pref
        return result, delta

    return inner


def compose_train_files(train_count: int, use_shuffled: bool) -> list[str]:
    prefix = "shuffle_train" if use_shuffled else "train"
    middle = f"of-{train_count}"
    surfix = "parquet"

    train_files = []
    if train_count > 1:
        just_size = 2
        for i in range(train_count):
            sub_file = f"{prefix}-{str(i).rjust(just_size, '0')}-{middle}.{surfix}"
            train_files.append(sub_file)
    else:
        train_files.append(f"{prefix}.{surfix}")

    return train_files


ONE_PERCENT = 0.01
NINETY_NINE_PERCENT = 0.99


def compose_gt_file(filters: float | str | None = None) -> str:
    if filters is None:
        return "neighbors.parquet"

    if filters == ONE_PERCENT:
        return "neighbors_head_1p.parquet"

    if filters == NINETY_NINE_PERCENT:
        return "neighbors_tail_1p.parquet"

    msg = f"Filters not supported: {filters}"
    raise ValueError(msg)


def download_file(url: str, dest_path: str, chunk_size: int = 8192, show_progress: bool = True) -> None:
    """Download a file from a URL to a local path, with progress bar. Skips if file exists."""
    if os.path.exists(dest_path):
        return
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get('content-length', 0))
    with open(dest_path, 'wb') as file, tqdm(
        desc=f"Downloading {os.path.basename(dest_path)}",
        total=total,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
        disable=not show_progress
    ) as bar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                file.write(chunk)
                bar.update(len(chunk))
