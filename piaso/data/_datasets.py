"""
Tutorial dataset management for PIASO.

Downloads and caches tutorial datasets from Zenodo (DOI: 10.5281/zenodo.19699639).
Registry is defined in datasets.json in the PIASO-data GitHub repository.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Union

from ._genome import PIASO_DATA_DIR

DATASETS_DIR = PIASO_DATA_DIR / "datasets"

# Bundled registry URL (fetched from PIASO-data GitHub repo at runtime)
_REGISTRY_URL = (
    "https://raw.githubusercontent.com/genecell/PIASO-data/master/datasets.json"
)

# Inline registry as fallback (updated at release time)
_BUILTIN_REGISTRY = None  # loaded lazily from _registry_builtin.json
_registry_cache = None


def _load_registry(force_refresh: bool = False) -> dict:
    """Load dataset registry, trying: memory cache → local cache → remote → builtin."""
    global _registry_cache
    if _registry_cache is not None and not force_refresh:
        return _registry_cache

    cache_path = PIASO_DATA_DIR / "datasets.json"

    # Try local cache (unless force refresh)
    if not force_refresh and cache_path.exists():
        try:
            with open(cache_path) as f:
                _registry_cache = json.load(f)
            return _registry_cache
        except (json.JSONDecodeError, OSError):
            pass

    # Try remote
    try:
        import requests
        resp = requests.get(_REGISTRY_URL, timeout=10)
        resp.raise_for_status()
        _registry_cache = resp.json()
        # Cache locally
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(resp.text)
        return _registry_cache
    except Exception:
        pass

    # Fallback: use builtin registry shipped with PIASO
    builtin_path = Path(__file__).parent / "_registry_builtin.json"
    if builtin_path.exists():
        with open(builtin_path) as f:
            _registry_cache = json.load(f)
        return _registry_cache

    # Last resort: use local cache even if we wanted refresh
    if cache_path.exists():
        with open(cache_path) as f:
            _registry_cache = json.load(f)
        return _registry_cache

    raise RuntimeError(
        "Could not load dataset registry. Check your internet connection, "
        "or manually place datasets.json in ~/.piaso/data/"
    )


def _md5_file(path: Path) -> str:
    """Compute MD5 checksum of a file."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_with_progress(url: str, dest: Path, expected_size: int = 0) -> None:
    """Download a file with a progress bar."""
    import urllib.request

    def _reporthook(block_num, block_size, total_size):
        if total_size <= 0:
            total_size = expected_size
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb_down = downloaded / 1e6
            mb_total = total_size / 1e6
            sys.stdout.write(f"\r  {mb_down:.1f} / {mb_total:.1f} MB ({pct}%)")
        else:
            sys.stdout.write(f"\r  {downloaded / 1e6:.1f} MB")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, str(dest), reporthook=_reporthook)
    sys.stdout.write("\n")
    sys.stdout.flush()


def list_datasets() -> Dict[str, dict]:
    """List available tutorial datasets.

    Returns
    -------
    datasets : dict
        Dictionary mapping dataset names to their metadata.

    Examples
    --------
    >>> piaso.data.list_datasets()
    """
    reg = _load_registry()
    datasets = reg.get("datasets", {})

    # Also print a formatted table
    print(f"{'Name':<30s} {'Title':<55s} {'Size':>10s}")
    print("-" * 97)
    for name, info in datasets.items():
        size_mb = info.get("size_bytes", 0) / 1e6
        title = info.get("title", "")
        if len(title) > 53:
            title = title[:50] + "..."
        print(f"  {name:<28s} {title:<55s} {size_mb:>8.1f} MB")

    return datasets


def dataset_info(name: str) -> dict:
    """Get metadata for a specific dataset.

    Parameters
    ----------
    name : str
        Dataset name (e.g., 'sea_ad_mtg_20k').

    Returns
    -------
    info : dict
        Dataset metadata including title, description, URL, size, etc.
    """
    reg = _load_registry()
    datasets = reg.get("datasets", {})
    if name not in datasets:
        available = ", ".join(datasets.keys())
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {available}"
        )
    return datasets[name]


def fetch_dataset(name: str, force: bool = False) -> Path:
    """Download a tutorial dataset to ~/.piaso/data/datasets/.

    Downloads the file if not already cached. Verifies MD5 checksum.

    Parameters
    ----------
    name : str
        Dataset name (e.g., 'sea_ad_mtg_20k', 'mouse_brain_10k_gemx').
    force : bool
        If True, re-download even if file exists and checksum matches.

    Returns
    -------
    path : Path
        Local path to the downloaded file.

    Examples
    --------
    >>> path = piaso.data.fetch_dataset("sea_ad_mtg_20k")
    >>> adata = anndata.read_h5ad(path)
    """
    info = dataset_info(name)
    filename = info["filename"]
    dest = DATASETS_DIR / filename

    # Check cache
    if not force and dest.exists():
        expected_md5 = info.get("md5", "")
        if expected_md5:
            actual_md5 = _md5_file(dest)
            if actual_md5 == expected_md5:
                return dest
            else:
                print(f"Checksum mismatch for '{name}', re-downloading...")
        else:
            return dest

    # Download
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    size_mb = info.get("size_bytes", 0) / 1e6
    print(f"Downloading '{info['title']}' ({size_mb:.1f} MB)...")
    _download_with_progress(info["url"], dest, info.get("size_bytes", 0))

    # Verify checksum
    expected_md5 = info.get("md5", "")
    if expected_md5:
        actual_md5 = _md5_file(dest)
        if actual_md5 != expected_md5:
            dest.unlink()
            raise RuntimeError(
                f"MD5 checksum mismatch for '{name}'.\n"
                f"Expected: {expected_md5}\n"
                f"Got:      {actual_md5}\n"
                f"The file may be corrupted. Try again with force=True."
            )

    print(f"Saved to {dest}")
    return dest


def load_dataset(name: str, return_type: str = "anndata",
                 modality: str = "RNA", cytome_path=None, backed: bool = True,
                 **kwargs):
    """Download and load a tutorial dataset as AnnData or a cytome Dataset.

    Parameters
    ----------
    name : str
        Dataset name (e.g., 'sea_ad_mtg_20k').
    return_type : str, default 'anndata'
        ``'anndata'`` returns an AnnData (default). ``'cytome'`` converts the
        downloaded data to a cytome and returns an **opened** ``cytome.Dataset``
        (the caller owns it — call ``.close()`` when done). The cytome is built
        once and cached, so repeat calls reopen it.
    modality : str, default 'RNA'
        Modality used when converting to a cytome (``return_type='cytome'``).
    cytome_path : str, optional
        Where to write/read the cytome. Defaults to a ``.cytome`` file beside the
        cached download (``<name>.cytome``).
    backed : bool, default True
        For h5ad → cytome, use the streaming (bounded-RAM) ``from_h5ad`` path.
    **kwargs
        Passed to ``anndata.read_h5ad()`` / ``scanpy.read_10x_h5()`` (AnnData
        path, or the 10x/csv → cytome conversion).

    Returns
    -------
    AnnData or cytome.Dataset

    Examples
    --------
    >>> adata = piaso.data.load_dataset("sea_ad_mtg_20k")
    >>> ds = piaso.data.load_dataset("sea_ad_mtg_20k", return_type="cytome")
    """
    if return_type not in ("anndata", "cytome"):
        raise ValueError(
            f"Invalid return_type={return_type!r}. Choose 'anndata' or 'cytome'."
        )
    info = dataset_info(name)
    path = fetch_dataset(name)
    fmt = info.get("format", "h5ad")

    def _read_anndata():
        if fmt == "h5ad":
            import anndata
            return anndata.read_h5ad(path, **kwargs)
        elif fmt == "10x_h5":
            import scanpy as sc
            return sc.read_10x_h5(str(path), **kwargs)
        elif fmt == "csv":
            import pandas as pd
            return pd.read_csv(path, **kwargs)
        raise ValueError(
            f"Unsupported format '{fmt}' for dataset '{name}'. "
            f"Use fetch_dataset() to get the file path and load manually."
        )

    if return_type == "anndata":
        return _read_anndata()

    # --- return_type == 'cytome' ---
    import cytome
    from pathlib import Path
    cpath = Path(cytome_path) if cytome_path is not None else Path(path).with_suffix(".cytome")
    if cpath.exists():
        return cytome.open(str(cpath))
    cpath.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "h5ad":
        # streaming conversion (bounded RAM); no in-memory AnnData
        return cytome.from_h5ad(str(path), output=str(cpath),
                                modality=modality, backed=backed)
    if fmt == "10x_h5":
        # native CellRanger .h5 reader — no AnnData intermediate
        return cytome.from_10x_h5(str(path), output=str(cpath))
    # csv (rare): build an AnnData first, then convert
    adata = _read_anndata()
    return cytome.from_anndata(adata, modality=modality, output=str(cpath))


def refresh_registry() -> None:
    """Force re-download the dataset registry from GitHub.

    Use this if new datasets have been added to PIASO-data.
    """
    _load_registry(force_refresh=True)
    print("Dataset registry refreshed.")
