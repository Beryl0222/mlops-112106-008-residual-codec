"""码本整版的不可变留存与容量管理。

一次发布产生一个新版本号，整版（所有组、所有层）原子可见。旧版本对
历史读者保持原样，绝不重解释已写出的码字。``max_versions`` 达到容量
上限时，只有最旧版本无任何未消费引用才允许剔除，否则由
:class:`RetentionExceeded` 报告仍在引用的片段，交上游判断何时继续。
"""

from dataclasses import dataclass

import numpy as np

from .errors import RetentionExceeded, UnknownCodebookVersion


@dataclass
class _Reference:
    segment_id: str
    group: str
    layers: int


class VersionStore:
    def __init__(self, initial_books: dict, max_versions: int):
        self.max_versions = int(max_versions)
        self._versions = {
            0: self._freeze(initial_books),
        }
        self._next_id = 1
        self._refs = {}  # version -> {segment_id: _Reference}

    @staticmethod
    def _freeze(books: dict) -> dict:
        return {
            group: [np.asarray(b, dtype=np.float32).copy() for b in layers]
            for group, layers in books.items()
        }

    @property
    def current_version(self) -> int:
        return self._next_id - 1

    def versions(self) -> list:
        return sorted(self._versions)

    def get(self, version: int) -> dict:
        try:
            return self._versions[int(version)]
        except KeyError:
            raise UnknownCodebookVersion("<unknown>", int(version))

    def current_books(self) -> dict:
        return self.get(self.current_version)

    def add_reference(self, segment_id: str, group: str, layers: int,
                      version: int | None = None) -> int:
        version = self.current_version if version is None else int(version)
        if version not in self._versions:
            raise UnknownCodebookVersion(segment_id, version)
        self._refs.setdefault(version, {})[segment_id] = _Reference(
            segment_id=segment_id, group=group, layers=int(layers),
        )
        return version

    def release_reference(self, segment_id: str) -> None:
        for holders in self._refs.values():
            holders.pop(segment_id, None)

    def references_for(self, version: int) -> list:
        return [
            {
                "segment_id": ref.segment_id,
                "group": ref.group,
                "layers": ref.layers,
            }
            for ref in self._refs.get(version, {}).values()
        ]

    def can_publish(self) -> None:
        """容量预检：不改变任何状态，不通过则抛 RetentionExceeded。"""
        if len(self._versions) >= self.max_versions:
            oldest = min(self._versions)
            held = self.references_for(oldest)
            if held:
                raise RetentionExceeded(
                    oldest, self.max_versions, held,
                )

    def publish(self, snapshot: dict) -> dict:
        """尝试发布新整版；容量不足且旧版仍被引用时拒绝并给出明细。"""
        self.can_publish()
        if len(self._versions) >= self.max_versions:
            oldest = min(self._versions)
            del self._versions[oldest]
            self._refs.pop(oldest, None)
        version = self._next_id
        self._versions[version] = self._freeze(snapshot)
        self._next_id += 1
        return {
            "version": version,
            "retained": self.versions(),
            "evicted": [v for v in range(version)
                        if v not in self._versions],
        }
