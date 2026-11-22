"""码本训练：提交事实累积、EMA 更新与冷码复活。

每个 (通道组, 层) 维护：

* ``hist_count``：码历史指派总数（跨发布持久保留）；
* ``sum_residual`` / ``assigned``：本步（两次发布之间）累计被分配的
  残差向量与次数，只接收已提交层的冻结事实；
* ``miss_streak``：连续无指派的更新次数，初值可由交换资料给定。

发布规则：指派数达到 ``min_assigned`` 的码按
``decay * 旧向量 + (1-decay) * 本步质心`` 更新；不足者保留原码并累加
未指派计数；达到 ``cold_after`` 的冷码用同组同层已提交残差中平方量化
误差最大者复活，平局按档案片段 ID、帧序裁决，每份残差同层至多复活一次。
"""

from dataclasses import dataclass, field

import numpy as np

from .errors import AlreadyCommitted, InconsistentSubmission


@dataclass
class LayerStats:
    hist_count: np.ndarray
    sum_residual: np.ndarray
    assigned: np.ndarray
    miss_streak: np.ndarray


@dataclass
class _PendingFact:
    segment_id: str
    group: str
    layer: int
    frame_offset: int
    indices: np.ndarray
    residuals: np.ndarray
    errors: np.ndarray


@dataclass
class LayerPublishReport:
    ema_codes: list = field(default_factory=list)
    frozen_codes: list = field(default_factory=list)
    revived: list = field(default_factory=list)
    cold_remaining: list = field(default_factory=list)
    assigned_counts: object = None


class CodebookTrainer:
    def __init__(self, books: dict, policy: dict, initial_cold=None):
        # 当前生效码本（可变工作副本；快照由版本库负责不可变留存）。
        self.books = {
            group: [np.asarray(b, dtype=np.float32).copy() for b in layers]
            for group, layers in books.items()
        }
        self.policy = dict(policy)
        self._stats = {}
        for group, layers in self.books.items():
            self._stats[group] = []
            for book in layers:
                k, d = book.shape
                self._stats[group].append(LayerStats(
                    hist_count=np.zeros(k, dtype=np.int64),
                    sum_residual=np.zeros((k, d), dtype=np.float64),
                    assigned=np.zeros(k, dtype=np.int64),
                    miss_streak=np.zeros(k, dtype=np.int64),
                ))
        for group, layer, code, count in initial_cold or []:
            self._stats[group][layer].miss_streak[code] = int(count)
        # key=(segment_id, layer)；同一提交重放直接拒绝。
        self._committed = set()
        self._pending = []

    # ------------------------------------------------------------------
    def commit(self, segment_id: str, group: str, layer: int,
               residuals: np.ndarray, indices: np.ndarray,
               errors: np.ndarray, frame_offset: int = 0) -> None:
        key = (segment_id, layer)
        if key in self._committed:
            raise AlreadyCommitted(segment_id, [layer])
        if group not in self.books or layer >= len(self.books[group]):
            raise InconsistentSubmission(
                f"组 {group!r} 第 {layer} 层不存在对应码本"
            )
        residuals = np.asarray(residuals, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int64)
        errors = np.asarray(errors, dtype=np.float64)
        if not (len(residuals) == len(indices) == len(errors)):
            raise InconsistentSubmission("提交事实行数不一致")
        stats = self._stats[group][layer]
        if len(indices):
            if indices.min() < 0 or indices.max() >= len(stats.assigned):
                raise InconsistentSubmission("码字编号超出码本范围")
            np.add.at(stats.assigned, indices, 1)
            np.add.at(stats.hist_count, indices, 1)
            np.add.at(stats.sum_residual, indices,
                      residuals.astype(np.float64))
        self._pending.append(_PendingFact(
            segment_id=segment_id, group=group, layer=layer,
            frame_offset=frame_offset,
            indices=indices.copy(), residuals=residuals.copy(),
            errors=errors.copy(),
        ))
        self._committed.add(key)

    def is_committed(self, segment_id: str, layer: int) -> bool:
        return (segment_id, layer) in self._committed

    # ------------------------------------------------------------------
    def publish(self) -> tuple[dict, dict]:
        """把全部已提交事实整版应用，返回 (新码本, 分组成层报告)。"""
        decay = float(self.policy["decay"])
        min_assigned = int(self.policy["min_assigned"])
        cold_after = int(self.policy["cold_after"])
        report = {}
        for group, layers in self.books.items():
            report[group] = []
            for layer, book in enumerate(layers):
                stats = self._stats[group][layer]
                new_book = book.copy()
                rep = LayerPublishReport(assigned_counts=stats.assigned.copy())
                k = len(book)
                # 1) EMA 更新与未指派计数。
                for code in range(k):
                    if stats.assigned[code] >= min_assigned:
                        centroid = stats.sum_residual[code] / stats.assigned[code]
                        new_book[code] = (
                            decay * book[code].astype(np.float64)
                            + (1.0 - decay) * centroid
                        ).astype(np.float32)
                        stats.miss_streak[code] = 0
                        rep.ema_codes.append(code)
                    else:
                        # 样本不足保留原码；本步无指派则累加冷度。
                        if stats.assigned[code] == 0:
                            stats.miss_streak[code] += 1
                        rep.frozen_codes.append(code)
                # 2) 冷码复活：候选必须来自同组同层的已提交残差。
                cold_codes = [
                    c for c in range(k)
                    if stats.assigned[c] == 0
                    and stats.miss_streak[c] >= cold_after
                ]
                pool = [
                    p for p in self._pending
                    if p.group == group and p.layer == layer
                ]
                candidates = []
                seen = set()
                for fact in pool:
                    for row in range(len(fact.indices)):
                        ident = (fact.segment_id,
                                 fact.frame_offset + row)
                        if ident in seen:
                            continue
                        seen.add(ident)
                        candidates.append((
                            -float(fact.errors[row]),
                            fact.segment_id,
                            fact.frame_offset + row,
                            fact.residuals[row],
                        ))
                # (-误差, 片段ID 字典序, 全局帧序)；误差最大且平局可裁决。
                candidates.sort(key=lambda item: (item[0], item[1], item[2]))
                used = 0
                for code in cold_codes:
                    if used >= len(candidates):
                        rep.cold_remaining.append(code)
                        continue
                    _, seg_id, frame_index, residual = candidates[used]
                    used += 1
                    new_book[code] = residual.astype(np.float32).copy()
                    stats.miss_streak[code] = 0
                    # 复活码不再归入纯冻结，三类结果互斥。
                    if code in rep.frozen_codes:
                        rep.frozen_codes.remove(code)
                    rep.revived.append({
                        "code": code,
                        "segment": seg_id,
                        "frame": frame_index,
                        "error": float(-candidates[used - 1][0]),
                    })
                # 3) 本步累加器清零，历史计数与冷度保留。
                stats.assigned[:] = 0
                stats.sum_residual[:] = 0.0
                self.books[group][layer] = new_book
                report[group].append(rep)
        self._pending.clear()
        snapshot = {
            group: [b.copy() for b in layers]
            for group, layers in self.books.items()
        }
        return snapshot, report
