"""One way to download a file into the PIASO data root.

Every fetcher in ``piaso.data`` used to call ``urllib.request.urlretrieve``
its own way. Some wrote straight to the final name, so an interrupted or
truncated download looked complete to the next call ("already available")
and was read as if it were whole (the genome BEDs and GTF, the TF list).
None had a timeout, so a stalled server hung the caller. ``fetch_dataset``
renamed a ``mkstemp`` file into place and so left every dataset with mode
0600 on a shared machine.

:func:`download` is the one rule: stream to a temporary file beside the
destination, check it (the server's length, an expected size, an MD5, or a
caller's own check), and only then rename it into place, with the
permissions a new file gets under the caller's umask. A failure leaves the
destination as it was. Network errors are retried with a growing pause.
"""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

#: seconds without data before a connection or read gives up
DEFAULT_TIMEOUT = 60.0
#: attempts per download (network errors only; a failed check is not retried)
DEFAULT_RETRIES = 3


class DownloadCheckError(ValueError):
    """The bytes arrived but are not the file asked for (size, MD5 or content)."""


class TruncatedTransfer(ConnectionError):
    """Fewer bytes arrived than the server announced: a network failure, retried."""


def _umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_once(url: str, tmp: Path, timeout: float, progress: bool, expected_size: int) -> None:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "piaso-data"})
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
        length = resp.headers.get("Content-Length")
        length = int(length) if length and length.isdigit() else None
        total = length or expected_size or 0
        got = 0
        while True:
            block = resp.read(1 << 20)
            if not block:
                break
            out.write(block)
            got += len(block)
            if progress:
                if total:
                    sys.stdout.write(f"\r  {got / 1e6:.1f} / {total / 1e6:.1f} MB "
                                     f"({min(100, got * 100 // total)}%)")
                else:
                    sys.stdout.write(f"\r  {got / 1e6:.1f} MB")
                sys.stdout.flush()
        if progress:
            sys.stdout.write("\n")
            sys.stdout.flush()
    if length is not None and got != length:
        raise TruncatedTransfer(f"{url}: received {got:,} of {length:,} bytes")


def fetch_with_retries(url: str, path, *, timeout: float = DEFAULT_TIMEOUT,
                       retries: int = DEFAULT_RETRIES, progress: bool = False,
                       expected_size: int = 0) -> None:
    """Write ``url``'s bytes to ``path`` (overwritten), retrying network errors.

    A transfer cut short (fewer bytes than the server announced), a timeout or
    a reset is retried; a client error (HTTP 4xx) is raised at once.
    """
    import http.client
    import urllib.error
    last = None
    for attempt in range(max(1, int(retries))):
        try:
            _fetch_once(url, Path(path), timeout, progress, int(expected_size or 0))
            return
        except DownloadCheckError:
            raise
        except urllib.error.HTTPError as exc:
            if exc.code < 500:
                raise
            last = exc
        except (OSError, http.client.HTTPException) as exc:   # URLError, timeouts, resets, IncompleteRead
            last = exc
        if attempt + 1 < max(1, int(retries)):
            time.sleep(2 ** attempt)
    raise last


def download(
    url: str,
    dest,
    *,
    md5: Optional[str] = None,
    expected_size: Optional[int] = None,
    check: Optional[Callable[[Path], None]] = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    progress: bool = False,
    remove_sidecars: bool = False,
) -> Path:
    """Download ``url`` to ``dest``, atomically.

    Parameters
    ----------
    md5, expected_size
        Checked on the temporary file before it is renamed into place.
    check
        Called with the temporary file's path; raise ``ValueError`` to refuse it
        (e.g. a host that answers with an error page and HTTP 200).
    timeout, retries
        Per-connection timeout in seconds, and attempts on a network error,
        with a pause of 1, 2, 4 ... seconds between them.
    remove_sidecars
        Delete ``dest``'s ``-wal`` / ``-shm`` / ``-journal`` before the rename.
        For SQLite files (a ``.cytome``): SQLite would replay an old ``-wal``
        onto the new file.

    Returns
    -------
    Path
        ``dest``. On any failure ``dest`` is unchanged and nothing is left behind.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".part", dir=str(dest.parent))
    os.close(fd)
    tmp = Path(tmp)
    try:
        fetch_with_retries(url, tmp, timeout=timeout, retries=retries, progress=progress,
                           expected_size=int(expected_size or 0))
        size = tmp.stat().st_size
        if expected_size and size != int(expected_size):
            raise DownloadCheckError(f"{url}: {size:,} bytes, expected {int(expected_size):,}")
        if md5:
            got = _md5(tmp)
            if got != md5:
                raise DownloadCheckError(f"{url}: MD5 {got}, expected {md5}")
        if check is not None:
            check(tmp)
        os.chmod(tmp, 0o666 & ~_umask())
        if remove_sidecars:
            for suffix in ("-wal", "-shm", "-journal"):
                try:
                    os.remove(f"{dest}{suffix}")
                except FileNotFoundError:
                    pass
        os.replace(tmp, dest)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dest


def copy_atomic(src, dest) -> Path:
    """Copy ``src`` to ``dest`` through a temporary file, so ``dest`` is never half-written."""
    import shutil
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{dest.name}.", suffix=".part", dir=str(dest.parent))
    os.close(fd)
    try:
        shutil.copy2(str(src), tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest
