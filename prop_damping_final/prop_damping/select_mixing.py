"""Selection and freeze-file helpers for the mixing study."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results" / "mixing"


def freeze(winners: list[dict], baselines: dict, path: Path | None = None) -> Path:
    path = path or RES / "selected_mixing.json"
    if path.exists():
        raise FileExistsError(f"freeze file already exists: {path}")
    payload = {"winners": winners, "baselines": baselines,
               "confirmatory_family": ["CTRL_lin0.9", "IRIE_exact", "rule_EW", "rule_WT",
                                       "MP2", "best_const_alpha_dev", "oracle:scalar"],
               "created_utc": datetime.now(timezone.utc).isoformat()}
    raw = json.dumps(payload, indent=2, sort_keys=True).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    (path.with_suffix(".sha256")).write_text(f"{digest}  {path.name}\n")
    return path
