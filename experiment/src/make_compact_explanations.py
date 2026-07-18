from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from improvement_common import IMPROVEMENT_DATA_DIR, ensure_improvement_dirs


def compact_value(frame: pd.DataFrame) -> pd.Series:
    obj = frame["structured_object"].fillna("").astype(str).str.strip()
    act = frame["structured_act"].fillna("").astype(str).str.strip()
    valid = frame["valid_structure"].fillna(False).astype(bool)
    value = (obj + "；" + act).where(valid, "")
    return value


def main() -> None:
    ensure_improvement_dirs()
    outputs: list[dict[str, object]] = []
    for source in sorted(IMPROVEMENT_DATA_DIR.glob("douyin_*_structured_v2.csv")):
        frame = pd.read_csv(source, encoding="utf-8-sig")
        if "structured_object" not in frame or "structured_act" not in frame:
            continue
        frame["qwen_background_compact"] = compact_value(frame)
        frame["compact_version"] = "object_act_v1"
        target = source.with_name(source.stem.replace("structured_v2", "compact_object_act_v1") + ".csv")
        frame.to_csv(target, index=False, encoding="utf-8-sig")
        outputs.append(
            {
                "source": str(source),
                "target": str(target),
                "rows": len(frame),
                "nonempty_compact": int(frame["qwen_background_compact"].ne("").sum()),
                "compact_sha256": hashlib.sha256(
                    "\n".join(frame["qwen_background_compact"].astype(str)).encode("utf-8")
                ).hexdigest(),
            }
        )
    manifest = {"representation": "object+act without generated evidence", "outputs": outputs}
    (IMPROVEMENT_DATA_DIR / "compact_representation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
