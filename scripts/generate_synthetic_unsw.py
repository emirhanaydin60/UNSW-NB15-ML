from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
]


def make_rows(per_class: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    idx = 0
    for cls in CLASSES:
        for _ in range(per_class):
            row = {
                "id": idx,
                "label": 0,
                "attack_cat": cls,
                "proto": rng.choice(["tcp", "udp", "icmp"]),
                "service": rng.choice(["http", "dns", "smtp", "ftp"]),
                "state": rng.choice(["FIN", "CON", "INT", "REQ"]),
            }
            for i in range(39):
                row[f"f{i}"] = float(rng.normal())
            rows.append(row)
            idx += 1
    return pd.DataFrame(rows)


def main() -> None:
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    train = make_rows(per_class=100, seed=42)
    test = make_rows(per_class=50, seed=43)

    train.to_csv(out_dir / "UNSW_NB15_training-set.csv", index=False)
    test.to_csv(out_dir / "UNSW_NB15_testing-set.csv", index=False)
    print("Synthetic datasets created")


if __name__ == "__main__":
    main()
