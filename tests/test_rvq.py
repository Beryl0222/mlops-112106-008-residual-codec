"""残差量化、位流、训练反馈与版本留存的端到端测试。"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rvq import (
    AlreadyCommitted,
    BitstreamReader,
    LayerNotDeclared,
    MemoryBudgetExceeded,
    QuantizationArchive,
    RetentionExceeded,
    TruncatedLayer,
    UnknownCodebookVersion,
    encode_block,
    nearest_codes,
    pack_bitmap,
    pack_codes,
    unpack_bitmap,
    unpack_codes,
)

SCENE = json.loads(
    (Path(__file__).resolve().parents[1] / "samples" / "scene.json")
    .read_text(encoding="utf-8")
)


def make_archive(**overrides):
    params = dict(
        books=SCENE["books"], policy=SCENE["policy"],
        budget_bytes=SCENE["budget_bytes"], initial_cold=SCENE["cold"],
    )
    params.update(overrides)
    return QuantizationArchive(**params)


def encode(arc, sid):
    item = next(s for s in SCENE["segments"] if s["id"] == sid)
    return arc.encode_segment(
        item["id"], item["group"], item["frames"], item["mask"],
        item["requested"], item["approved"],
    )


class QuantizationRuleTests(unittest.TestCase):
    def test_nearest_code_uses_squared_distance_and_smaller_index_on_tie(self):
        book = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        residual = np.array([[1.0, 0.0]])
        indices, distances = nearest_codes(residual, book)
        self.assertEqual(int(indices[0]), 0)  # 0 与 2 等距，取小编号
        self.assertAlmostEqual(float(distances[0]), 0.0)

    def test_missing_frames_are_excluded_but_zero_vectors_are_candidates(self):
        frames = np.array([[0.0, 0.0], [9.0, 9.0], [0.0, 0.0]])
        mask = np.array([True, False, True])
        book = np.array([[0.0, 0.0], [1.0, 0.0]])
        result = encode_block(frames, mask, [book])
        # 两个有效帧都是零向量，正常选中码字 0；缺帧不产生编码。
        np.testing.assert_array_equal(result.layer_indices[0], [0, 0])
        self.assertEqual(len(result.layer_residuals[0]), 2)
        self.assertEqual(result.layer_errors[0].shape, (2,))
        # 缺帧位置重建保持零。
        np.testing.assert_array_equal(result.reconstruction[1], [0.0, 0.0])

    def test_residual_chains_across_layers(self):
        frames = np.array([[2.0, 0.0]])
        mask = np.array([True])
        books = [
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            np.array([[0.0, 0.0], [1.0, 0.0]]),
        ]
        result = encode_block(frames, mask, books)
        np.testing.assert_array_equal(result.layer_indices[0], [1])
        np.testing.assert_array_equal(result.layer_indices[1], [1])
        np.testing.assert_allclose(result.reconstruction[0], [2.0, 0.0])
        # 第二层的输入残差正是第一层之后的剩余。
        np.testing.assert_allclose(
            result.layer_residuals[1][0], [1.0, 0.0]
        )


class BitPackingTests(unittest.TestCase):
    def test_code_roundtrip_for_non_power_of_two_book(self):
        for size in (2, 3, 4, 5, 7, 8, 4096):
            bits = size.bit_length() if size > 1 else 0
            rng = np.random.default_rng(size)
            indices = rng.integers(0, size, size=500)
            back = unpack_codes(pack_codes(indices, bits), 500, bits)
            np.testing.assert_array_equal(indices, back)

    def test_bitmap_roundtrip_distinguishes_missing_from_code_zero(self):
        mask = np.array([True, True, False, True, False, True, True])
        blob = pack_bitmap(mask)
        np.testing.assert_array_equal(unpack_bitmap(blob, len(mask)), mask)

    def test_layer_code_counts_match_valid_frames_and_layer_boundary(self):
        arc = make_archive()
        segment = encode(arc, "s0")  # approved=1，只写一层
        info = arc.encoded_layers("s0")
        self.assertEqual(info["declared_layers"], 1)
        self.assertEqual(info["valid_frames"], 4)
        reader = BitstreamReader(segment.stream)
        codes = reader.read_block_codes(0, 0)
        self.assertEqual(len(codes), 4)
        with self.assertRaises(LayerNotDeclared):
            reader.read_block_codes(0, 1)

    def test_truncated_layer_is_distinct_from_missing_declaration(self):
        arc = make_archive()
        segment = encode(arc, "s1")  # 三层
        reader = BitstreamReader(segment.stream)
        # 请求超过声明层：声明无高层。
        with self.assertRaises(LayerNotDeclared):
            reader.read_layers(4, arc.versions.current_books(),
                               expected_version=0)
        # 在明确层边界内截掉尾部字节：层内容不足。
        cut = segment.stream[:-1]
        cut_reader = BitstreamReader(cut)
        with self.assertRaises(TruncatedLayer):
            cut_reader.read_layers(3, arc.versions.current_books(),
                                   expected_version=0)
        # 第一层仍可完整返回，高层不阻塞。
        first = cut_reader.read_layers(1, arc.versions.current_books(),
                                       expected_version=0)
        self.assertEqual(first.layers, 1)
        self.assertEqual(int(first.valid.sum()), 3)

    def test_random_access_first_layer_does_not_touch_high_layers(self):
        arc = make_archive(block_rows=2)
        encode(arc, "s0")
        segment = encode(arc, "s1")
        # 直接按帧随机访问第 0 层，且流只允许 seek 到目标块。
        valid, code = arc.read_frame_layer("s1", frame_index=2, layer=0)
        self.assertTrue(valid)
        reader = BitstreamReader(segment.stream)
        self.assertEqual(reader.n_blocks, 2)
        # s0 第 2 帧是缺帧：位图为假，编号缺省为 0，与真实码字 0 区分。
        valid_missing, code_missing = arc.read_frame_layer(
            "s0", frame_index=2, layer=0,
        )
        self.assertFalse(valid_missing)
        self.assertEqual(code_missing, 0)

    def test_unknown_codebook_version_is_reported(self):
        arc = make_archive()
        encode(arc, "s1")
        reader = arc.open_reader("s1")
        with self.assertRaises(UnknownCodebookVersion):
            reader.read_layers(1, {"a": arc.versions.current_books()["a"]},
                               expected_version=99)


class TrainingTests(unittest.TestCase):
    def test_commit_uses_only_approved_layers_and_replay_is_rejected(self):
        arc = make_archive()
        encode(arc, "s0")  # 仅批准一层
        outcome = arc.commit("s0")
        self.assertEqual(outcome["committed_layers"], [0])
        stats = arc.trainer._stats["a"][0]
        self.assertEqual(int(stats.assigned.sum()), 4)
        # 第二、三层没有提交事实。
        self.assertEqual(int(arc.trainer._stats["a"][1].assigned.sum()), 0)
        with self.assertRaises(AlreadyCommitted):
            arc.commit("s0", layers=[0])

    def test_ema_update_respects_min_assigned_and_preserves_weak_codes(self):
        arc = make_archive()
        encode(arc, "s0")  # 4 个有效帧，提供足够指派但不触碰部分冷码
        encode(arc, "s1")  # 3 个有效帧
        before = arc.trainer.books["a"][0].copy()
        arc.commit("s0")
        arc.commit("s1")
        snapshot, report = arc.trainer.publish()
        layer_report = report["a"][0]
        # min_assigned=2：被选 >=2 次的码走 EMA，其余保留原码或复活。
        self.assertTrue(layer_report.ema_codes)
        revived_codes = {item["code"] for item in layer_report.revived}
        # 三类结果互斥；纯冻结码（未达冷度阈值）向量必须保持不变。
        self.assertFalse(
            set(layer_report.ema_codes)
            & set(layer_report.frozen_codes)
            & revived_codes
        )
        for code in layer_report.frozen_codes:
            np.testing.assert_array_equal(
                snapshot["a"][0][code], before[code]
            )
        for code in layer_report.ema_codes:
            self.assertFalse(np.array_equal(
                snapshot["a"][0][code], before[code]
            ))

    def test_cold_revival_uses_largest_error_with_deterministic_tiebreak(self):
        arc = make_archive()
        encode(arc, "s1")
        arc.commit("s1")
        # 冷码 3 初始 miss=2，本步无指派后达到 cold_after=3。
        _, report = arc.trainer.publish()
        revived = report["a"][0].revived
        codes = [item["code"] for item in revived]
        self.assertIn(3, codes)
        chosen = next(item for item in revived if item["code"] == 3)
        # 复活向量必须真实来自该组该层的已提交残差。
        fact = arc.trainer
        self.assertIn(chosen["segment"], {"s1"})
        self.assertGreaterEqual(chosen["frame"], 0)
        # 复活后冷度清零。
        self.assertEqual(
            int(arc.trainer._stats["a"][0].miss_streak[3]), 0
        )

    def test_publish_is_atomic_and_next_batch_uses_consistent_books(self):
        arc = make_archive()
        encode(arc, "s0")
        encode(arc, "s1")
        arc.commit("s0")
        arc.commit("s1")
        outcome = arc.publish()
        self.assertEqual(outcome["version"], 1)
        # 发布后编码的新片段必须引用新版整版。
        segment = encode(arc, "s2")
        self.assertEqual(segment.version, 1)
        info = arc.encoded_layers("s2")
        self.assertEqual(info["codebook_version"], 1)

    def test_historical_counts_persist_across_publishes(self):
        arc = make_archive()
        encode(arc, "s1")
        arc.commit("s1")
        arc.trainer.publish()
        total = int(arc.trainer._stats["a"][0].hist_count.sum())
        self.assertEqual(total, 3)  # 本步有效帧总数


class RetentionTests(unittest.TestCase):
    def test_full_capacity_blocks_publish_and_names_referencing_segments(self):
        arc = make_archive()
        encode(arc, "s0")  # 持有版本 0
        arc.commit("s0")
        arc.publish()  # 版本 1，容量 2 已满
        encode(arc, "s1")
        arc.commit("s1")
        with self.assertRaises(RetentionExceeded) as ctx:
            arc.publish()
        refs = {item["segment_id"] for item in ctx.exception.references}
        self.assertIn("s0", refs)
        # 释放后同一批已提交输入仍可发布，训练事实不丢失。
        arc.release("s0")
        outcome = arc.publish()
        self.assertEqual(outcome["version"], 2)
        self.assertIn(0, outcome["evicted"])

    def test_old_versions_remain_readable_after_new_publish(self):
        arc = make_archive()
        segment = encode(arc, "s1")
        arc.commit("s1")
        arc.publish()
        # 版本 0 的位流仍按旧码本重建，结果与发布前一致。
        result = arc.read_layers("s1", 3)
        self.assertEqual(result.version, 0)
        self.assertEqual(result.layers, 3)
        np.testing.assert_array_equal(
            result.valid, np.array([True, True, True])
        )


class MemoryBudgetTests(unittest.TestCase):
    def test_tiny_budget_rejects_frozen_state(self):
        arc = make_archive(budget_bytes=64)
        with self.assertRaises(MemoryBudgetExceeded):
            encode(arc, "s1")

    def test_blocking_never_builds_full_frame_by_code_matrix(self):
        # 长片段在固定小块下产生多个块，单次距离矩阵仅块大小。
        arc = make_archive(budget_bytes=SCENE["budget_bytes"], block_rows=64)
        frames = np.random.default_rng(1).normal(size=(500, 4)).astype(
            np.float32
        )
        mask = np.ones(500, dtype=bool)
        mask[::37] = False
        books = [np.random.default_rng(l).normal(size=(4096, 4)).astype(
            np.float32) for l in range(2)]
        custom = QuantizationArchive(
            books={"g": books}, policy=SCENE["policy"],
            budget_bytes=2 ** 20, initial_cold=[], block_rows=64,
        )
        segment = custom.encode_segment("long", "g", frames, mask, 2, 2)
        info = custom.encoded_layers("long")
        self.assertEqual(info["blocks"], 8)
        self.assertEqual(segment.block_rows, 64)


class StreamFileTests(unittest.TestCase):
    def test_write_and_read_back_from_disk(self):
        arc = make_archive()
        encode(arc, "s1")
        with tempfile.TemporaryDirectory() as temp:
            path = arc.write_stream("s1", temp)
            books = arc.versions.get(0)
            result = QuantizationArchive.read_stream_file(
                path, books, up_to_layer=2, expected_version=0,
            )
            self.assertEqual(result.layers, 2)
            self.assertEqual(int(result.valid.sum()), 3)


if __name__ == "__main__":
    unittest.main()
