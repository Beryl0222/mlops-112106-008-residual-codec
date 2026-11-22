"""统一门面：编码、位流读写、提交训练、版本发布与释放。

调用方给出片段帧块、通道组、精度层数与当前码本即可编码；位流按获准
交付的层数写入，层边界与各层实际编码数量一致。训练侧只接收已提交层
冻结的分配与残差；发布为整版原子可见；释放由调用方在消费完成后发起。
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bitstream import BitstreamReader, BitstreamWriter, FrameLayers, bits_for_codes
from .budget import MemoryBudget
from .errors import (
    AlreadyCommitted,
    InconsistentSubmission,
    LayerNotDeclared,
)
from .quantize import encode_block, iter_block_ranges
from .training import CodebookTrainer
from .versions import VersionStore


@dataclass
class LayerFrozen:
    """一次编码冻结的单层事实（仅获准交付层才存在）。"""

    residuals: np.ndarray   # 全片段有效帧的该层输入残差
    indices: np.ndarray     # 全片段有效帧的码字编号
    errors: np.ndarray      # 全片段有效帧的平方量化误差


@dataclass
class EncodedSegment:
    id: str
    group: str
    requested: int
    approved: int
    version: int
    block_rows: int
    valid_mask: np.ndarray
    frozen: list
    stream: bytes = b""
    committed: set = field(default_factory=set)
    released: bool = False


class QuantizationArchive:
    def __init__(self, books: dict, policy: dict, budget_bytes: int,
                 initial_cold=None, block_rows: int | None = None):
        self.policy = dict(policy)
        self.budget = MemoryBudget(int(budget_bytes))
        self.trainer = CodebookTrainer(books, policy, initial_cold)
        self.versions = VersionStore(self.trainer.books,
                                     int(policy["max_versions"]))
        self._segments = {}
        self._fixed_block_rows = block_rows

    # ------------------------------------------------------------------
    def encode_segment(self, segment_id: str, group: str, frames,
                       mask, requested: int, approved: int) -> EncodedSegment:
        if segment_id in self._segments:
            raise InconsistentSubmission(f"片段 {segment_id!r} 已存在")
        if group not in self.trainer.books:
            raise InconsistentSubmission(f"未知通道组 {group!r}")
        frames = np.asarray(frames, dtype=np.float32)
        mask = np.asarray(mask, dtype=bool)
        if frames.shape[0] != mask.shape[0]:
            raise InconsistentSubmission("帧数与掩码长度不一致")
        group_books = self.trainer.books[group]
        if not (0 <= approved <= requested <= len(group_books)):
            raise InconsistentSubmission("精度层数超出该组可用码本")
        n_frames, dim = frames.shape
        max_codes = max(len(b) for b in group_books[:requested])
        if self._fixed_block_rows is None:
            block_rows = self.budget.fit_block_rows(dim, max_codes)
        else:
            block_rows = int(self._fixed_block_rows)
        block_rows = min(block_rows, max(1, n_frames))
        approved_books = group_books[:approved]
        code_bits = [bits_for_codes(len(b)) for b in approved_books]
        valid_total = int(mask.sum())
        # 先按待交付冻结状态预留预算，再实际分配；不足则不落任何片段状态。
        # 位流归档字节与一次输入块不计入预算。
        self.budget.reserve(
            self._frozen_bytes(n_frames, valid_total, approved),
            detail=f"片段 {segment_id!r} 冻结状态",
        )
        writer = BitstreamWriter(
            segment_id, group, self.versions.current_version,
            n_frames, block_rows, code_bits,
        )
        frozen = [LayerFrozen(
            residuals=np.zeros((valid_total, dim), dtype=np.float32),
            indices=np.zeros(valid_total, dtype=np.int64),
            errors=np.zeros(valid_total, dtype=np.float64),
        ) for _ in range(approved)]
        valid_full = np.zeros(n_frames, dtype=bool)
        global_cursor = 0
        for start, end in iter_block_ranges(n_frames, block_rows):
            block_frames = frames[start:end]
            block_mask = mask[start:end]
            result = encode_block(block_frames, block_mask, approved_books)
            block_layers = []
            valid_count = int(block_mask.sum())
            for layer in range(approved):
                idx = result.layer_indices[layer]
                block_layers.append(idx)
                slot = frozen[layer]
                slot.indices[global_cursor:global_cursor + valid_count] = idx
                slot.residuals[global_cursor:global_cursor + valid_count] = \
                    result.layer_residuals[layer]
                slot.errors[global_cursor:global_cursor + valid_count] = \
                    result.layer_errors[layer]
            writer.add_block(block_mask, block_layers)
            valid_full[start:end] = block_mask
            global_cursor += valid_count
        stream = writer.build()
        version = self.versions.add_reference(
            segment_id, group, approved,
        )
        segment = EncodedSegment(
            id=segment_id, group=group, requested=requested,
            approved=approved, version=version, block_rows=block_rows,
            valid_mask=valid_full, frozen=frozen, stream=stream,
        )
        self._segments[segment_id] = segment
        return segment

    def encoded_layers(self, segment_id: str) -> dict:
        """设备调用方读回实际写入的不同层及其版本、块边界。"""
        segment = self._segments[segment_id]
        reader = BitstreamReader(segment.stream)
        return {
            "segment": segment_id,
            "group": segment.group,
            "declared_layers": reader.declared_layers,
            "approved": segment.approved,
            "requested": segment.requested,
            "codebook_version": segment.version,
            "blocks": reader.n_blocks,
            "block_rows": reader.header["block_rows"],
            "code_bits": reader.header["code_bits"],
            "valid_frames": int(segment.valid_mask.sum()),
            "total_frames": len(segment.valid_mask),
        }

    # ------------------------------------------------------------------
    def commit(self, segment_id: str, layers=None) -> dict:
        """把已提交层的冻结事实送入训练；重放同一提交不重复学习。"""
        segment = self._require(segment_id)
        if segment.released:
            raise InconsistentSubmission(f"片段 {segment_id!r} 已释放")
        wanted = range(segment.approved) if layers is None else \
            [int(l) for l in layers]
        wanted = sorted(set(wanted))
        replay = [l for l in wanted
                  if l in segment.committed]
        if replay:
            raise AlreadyCommitted(segment_id, replay)
        frame_offset = 0
        learned = []
        for layer in wanted:
            if layer >= segment.approved:
                raise LayerNotDeclared(
                    segment_id, layer + 1, segment.approved
                )
            fact = segment.frozen[layer]
            self.trainer.commit(
                segment_id=segment_id, group=segment.group, layer=layer,
                residuals=fact.residuals, indices=fact.indices,
                errors=fact.errors, frame_offset=frame_offset,
            )
            segment.committed.add(layer)
            learned.append(layer)
        return {"segment": segment_id, "committed_layers": learned}

    def publish(self) -> dict:
        """整版发布：先做留存容量预检，再应用训练，最后原子换版。"""
        self.versions.can_publish()
        snapshot, report = self.trainer.publish()
        outcome = self.versions.publish(snapshot)
        return {
            "version": outcome["version"],
            "retained": outcome["retained"],
            "evicted": outcome["evicted"],
            "report": report,
        }

    @staticmethod
    def _frozen_bytes(n_frames: int, valid_total: int,
                      approved: int) -> int:
        """冻结状态在预算内的统一字节口径。"""
        return (
            n_frames * np.bool_().nbytes
            + approved * (
                valid_total * np.float32().nbytes
                + valid_total * np.int64().nbytes
                + valid_total * np.float64().nbytes
            )
        )

    def release(self, segment_id: str) -> dict:
        """消费完成后释放：解除版本引用并回收预算内的冻结状态。"""
        segment = self._require(segment_id)
        if not segment.released:
            self.versions.release_reference(segment_id)
            self.budget.release(self._frozen_bytes(
                len(segment.valid_mask),
                int(segment.valid_mask.sum()),
                segment.approved,
            ))
            segment.released = True
            segment.frozen = []
        return {"segment": segment_id, "released": True,
                "budget_used": self.budget.used_bytes}

    # ------------------------------------------------------------------
    def open_reader(self, segment_id: str) -> BitstreamReader:
        segment = self._require(segment_id)
        return BitstreamReader(segment.stream)

    def read_layers(self, segment_id: str, up_to_layer: int) -> FrameLayers:
        """按片段冻结的码本整版重建；缺层、截断、版本缺失分别报错。"""
        segment = self._require(segment_id)
        reader = BitstreamReader(segment.stream)
        books = self.versions.get(segment.version)
        return reader.read_layers(
            up_to_layer, books, expected_version=segment.version,
        )

    def read_frame_layer(self, segment_id: str, frame_index: int,
                         layer: int) -> tuple[bool, int]:
        """第一层随机访问入口：只 seek 目标块，不触碰未读高层。"""
        segment = self._require(segment_id)
        reader = BitstreamReader(segment.stream)
        block = frame_index // reader.header["block_rows"]
        row = frame_index % reader.header["block_rows"]
        return reader.read_frame_code(block, row, layer)

    def write_stream(self, segment_id: str, directory) -> Path:
        segment = self._require(segment_id)
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{segment_id}.rvq"
        path.write_bytes(segment.stream)
        return path

    @staticmethod
    def read_stream_file(path, books_by_group, up_to_layer: int,
                         expected_version: int) -> FrameLayers:
        """直接从磁盘文件读取，按给定版本码本重建。"""
        reader = BitstreamReader(Path(path).read_bytes())
        return reader.read_layers(
            up_to_layer, books_by_group,
            expected_version=expected_version,
        )

    def _require(self, segment_id: str) -> EncodedSegment:
        try:
            return self._segments[segment_id]
        except KeyError:
            raise InconsistentSubmission(
                f"片段 {segment_id!r} 尚未编码或不存在"
            ) from None
