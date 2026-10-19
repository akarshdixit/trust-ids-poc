"""
Fetch the IoT-23 labelled flow logs on demand (about 3 MB in total).

Used where the dataset cannot be shipped with the code, e.g. Streamlit Community
Cloud: the first time a live re-run or the live experiment needs IoT-23, the
three conn.log.labeled files are downloaded from the dataset's official host
(Stratosphere Lab, CTU) into data/iot23/, or into a temp directory if the
checkout is read-only. N-BaIoT is NOT fetched: it is ~300 MB and the attack
files come only inside .rar archives.

Downloads are treated as data only: each file must look like a Zeek log
(#separator / #fields headers) and is moved into place only when complete.
"""
import os
import tempfile
import urllib.request

from data import load_iot23 as io

IOT23_URL = ("https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios/"
             "{capture}/bro/conn.log.labeled")
IOT23_CAPTURES = (io.MIRAI, io.BENIGN_B, "CTU-Honeypot-Capture-4-1")
MAX_BYTES = 50 * 1024 * 1024  # the three logs are ~3 MB; refuse anything absurd


def _writable(path):
    try:
        os.makedirs(path, exist_ok=True)
        probe = os.path.join(path, ".write_test")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
        return True
    except OSError:
        return False


def _download(url, dest, timeout):
    tmp = dest + ".part"
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        size = 0
        while chunk := r.read(1 << 16):
            size += len(chunk)
            if size > MAX_BYTES:
                raise ValueError(f"{url}: larger than {MAX_BYTES} bytes, refusing")
            f.write(chunk)
    with open(tmp, encoding="utf-8", errors="replace") as f:
        head = f.read(4096)
    if not head.startswith("#separator") or "#fields" not in head:
        os.remove(tmp)
        raise ValueError(f"{url}: not a Zeek conn log")
    os.replace(tmp, dest)


def ensure_iot23(timeout=60, log=None):
    """Make the IoT-23 captures available; returns True if they are.

    Points data.load_iot23.DEFAULT_ROOT at wherever the files end up.
    """
    if io.available():
        return True
    root = io.DEFAULT_ROOT if _writable(io.DEFAULT_ROOT) else os.path.join(
        tempfile.gettempdir(), "trust_ids_iot23")
    for capture in IOT23_CAPTURES:
        dest = os.path.join(root, capture, "conn.log.labeled")
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if log:
            log(f"downloading IoT-23 {capture} ...")
        try:
            _download(IOT23_URL.format(capture=capture), dest, timeout)
        except (OSError, ValueError) as e:
            if log:
                log(f"could not fetch {capture}: {e}")
            if capture in (io.MIRAI, io.BENIGN_B):
                return False
    io.DEFAULT_ROOT = root
    io._CACHE.clear()
    return io.available()
