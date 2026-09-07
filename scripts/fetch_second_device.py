"""
Fetch a SECOND device's benign_traffic.csv — required for Scenario 1.

Why a second device is necessary (this is a measured result, not a preference):
Danmini_Doorbell's own benign_traffic.csv is stationary. Standardised distance
between decile block 1 and block 10 is 0.16, while consecutive blocks wander
0.30-0.74 — the endpoints are CLOSER together than neighbouring blocks are.
Splitting that file in half therefore yields no detectable drift: ADWIN never
fires, no update is ever proposed, and all four arms trace identical lines.

Ennio_Doorbell is chosen because it is the same DEVICE TYPE (doorbell) as
Danmini. That makes the Scenario-1 story clean and defensible: the traffic is
real, benign and clean-reputation at both endpoints, and the shift between them
is a genuine benign distribution change of the kind a firmware update or a
device swap produces — not something we invented.

Usage:
    python scripts/fetch_second_device.py [device_name]
"""
import os
import sys
import urllib.request

DEVICES = [
    "Ennio_Doorbell",                        # same type as Danmini — default
    "Ecobee_Thermostat",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
]

URL = ("https://archive.ics.uci.edu/ml/machine-learning-databases/00442/"
       "{device}/benign_traffic.csv")

BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "nbaiot")


def fetch(device):
    dest_dir = os.path.join(BASE, device)
    dest = os.path.join(dest_dir, "benign_traffic.csv")
    if os.path.exists(dest):
        print(f"already present: {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return dest

    os.makedirs(dest_dir, exist_ok=True)
    url = URL.format(device=device)
    print(f"downloading {url}")

    tmp = dest + ".part"

    def progress(blocks, block_size, total):
        if total > 0:
            pct = min(100.0, 100.0 * blocks * block_size / total)
            sys.stdout.write(f"\r  {pct:5.1f}%  ({total / 1e6:.1f} MB)")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    print()

    # Sanity-check before committing the file into place: it must be a CSV with
    # the same 115-column N-BaIoT header, or it is a soft-404 page.
    with open(tmp, "r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip()
    ncols = len(header.split(","))
    if ncols != 115 or not header.startswith("MI_dir_L5_weight"):
        os.remove(tmp)
        raise RuntimeError(
            f"downloaded file is not N-BaIoT data (header had {ncols} columns): "
            f"{header[:120]}"
        )

    os.replace(tmp, dest)
    print(f"wrote {dest} ({os.path.getsize(dest) / 1e6:.1f} MB, {ncols} columns)")
    return dest


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else DEVICES[0]
    path = fetch(device)
    print(f"\nNow run:\n  python experiments/run_scenarios.py --benign-b \"{path}\"")
