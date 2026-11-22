"""按交换资料的时间线驱动编码、读取、提交、发布与释放。

用法::

    python3 run_scene.py samples/scene.json
    python3 run_scene.py samples/retained-books.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from rvq import QuantizationArchive
from rvq.errors import (
    LayerNotDeclared,
    RetentionExceeded,
    RvqError,
    TruncatedLayer,
    UnknownCodebookVersion,
)


def _segments_by_id(scene: dict) -> dict:
    return {item["id"]: item for item in scene["segments"]}


def run(scene: dict) -> dict:
    archive = QuantizationArchive(
        books=scene["books"],
        policy=scene["policy"],
        budget_bytes=scene["budget_bytes"],
        initial_cold=scene.get("cold", []),
    )
    segments = _segments_by_id(scene)
    report = {"events": [], "reads": {}, "publishes": []}
    for event in scene["timeline"]:
        kind = event[0]
        if kind == "encode":
            item = segments[event[1]]
            encoded = archive.encode_segment(
                segment_id=item["id"], group=item["group"],
                frames=item["frames"], mask=item["mask"],
                requested=item["requested"], approved=item["approved"],
            )
            report["events"].append({
                "event": "encode", "segment": item["id"],
                "written_layers": encoded.approved,
                "version": encoded.version,
                "blocks": archive.encoded_layers(item["id"])["blocks"],
            })
        elif kind == "read_layers":
            _, segment_id, layers = event
            try:
                result = archive.read_layers(segment_id, layers)
            except (LayerNotDeclared, TruncatedLayer,
                    UnknownCodebookVersion) as exc:
                report["reads"][f"{segment_id}:{layers}"] = {
                    "outcome": type(exc).__name__,
                    "detail": str(exc),
                }
                report["events"].append(event)
                continue
            report["reads"][f"{segment_id}:{layers}"] = {
                "outcome": "ok",
                "frames": int(result.valid.sum()),
                "layers": result.layers,
                "version": result.version,
                "reconstruction_norm": float(
                    np.linalg.norm(result.reconstruction)
                ),
            }
            report["events"].append(event)
        elif kind == "commit":
            outcome = archive.commit(event[1])
            report["events"].append({"event": "commit", **outcome})
        elif kind == "publish":
            try:
                outcome = archive.publish()
            except RetentionExceeded as exc:
                report["events"].append({
                    "event": "publish", "blocked": True,
                    "version": exc.version,
                    "references": exc.references,
                })
                continue
            report["publishes"].append(outcome["version"])
            report["events"].append({
                "event": "publish", "version": outcome["version"],
                "retained": outcome["retained"],
                "evicted": outcome["evicted"],
            })
        elif kind == "release":
            outcome = archive.release(event[1])
            report["events"].append({"event": "release", **outcome})
        else:
            raise RvqError(f"未知时间线事件 {event!r}")
    report["budget_used_bytes"] = archive.budget.used_bytes
    report["versions"] = archive.versions.versions()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="驱动残差量化场景时间线")
    parser.add_argument("scene", type=Path)
    args = parser.parse_args()
    scene = json.loads(args.scene.read_text(encoding="utf-8"))
    print(json.dumps(run(scene), ensure_ascii=False, indent=2,
                     default=_json_default))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


if __name__ == "__main__":
    main()
