"""分块位流的写入与读取。

文件布局::

    [8 字节大端的 JSON 头长度][UTF-8 JSON 头][二进制负载]

头中的 ``offset`` 为相对二进制负载起点的字节偏移，读取时加上解析得到的
负载起点即可定位，因此读第一层时可以直接 seek 到对应 (块, 层) 区域，
不必触碰任何高层字节，也不会因 JSON 头长度变化而失准。

每块的有效帧位图按时序、低有效位在先打包；码字索引同样低有效位在先、
小端组位连续写入，层末仅做字节补齐。缺帧靠位图与码字 0 区分——
标记为零向量的正常帧仍会占据一个码字位置。
"""

from dataclasses import dataclass

import numpy as np

from .errors import LayerNotDeclared, TruncatedLayer, UnknownCodebookVersion

_MAGIC = b"RVQ1"


def bits_for_codes(codebook_size: int) -> int:
    """容纳编号 ``0..codebook_size-1`` 所需比特数。"""
    if codebook_size <= 1:
        return 0
    return int(codebook_size).bit_length()


def pack_bitmap(valid: np.ndarray) -> bytes:
    """布尔时序打包为字节，第 i 帧落在 ``byte[i//8]`` 的 ``i%8`` 位。"""
    valid = np.asarray(valid, dtype=bool)
    n = len(valid)
    out = bytearray((n + 7) // 8)
    for i, flag in enumerate(valid):
        if flag:
            out[i // 8] |= 1 << (i % 8)
    return bytes(out)


def unpack_bitmap(buf: bytes, n_rows: int) -> np.ndarray:
    out = np.zeros(n_rows, dtype=bool)
    for i in range(n_rows):
        if i // 8 < len(buf) and (buf[i // 8] >> (i % 8)) & 1:
            out[i] = True
    return out


def pack_codes(indices: np.ndarray, bits: int) -> bytes:
    """整数索引按低有效位在先连续打包，``bits=0`` 时输出空负载。"""
    if bits == 0 or len(indices) == 0:
        return b""
    indices = np.asarray(indices, dtype=np.int64)
    if np.any(indices >= (1 << bits)) or np.any(indices < 0):
        raise ValueError("码字编号超出该层声明的比特位数")
    total_bits = len(indices) * bits
    out = bytearray((total_bits + 7) // 8)
    bit_cursor = 0
    for value in indices.tolist():
        for b in range(bits):
            if (value >> b) & 1:
                out[bit_cursor // 8] |= 1 << (bit_cursor % 8)
            bit_cursor += 1
    return bytes(out)


def unpack_codes(buf: bytes, count: int, bits: int) -> np.ndarray:
    out = np.zeros(count, dtype=np.int64)
    if bits == 0 or count == 0:
        return out
    bit_cursor = 0
    for i in range(count):
        value = 0
        for b in range(bits):
            byte_index = bit_cursor // 8
            bit_index = bit_cursor % 8
            if byte_index >= len(buf):
                raise TruncatedLayer("", -1, -1, len(buf), byte_index + 1)
            if (buf[byte_index] >> bit_index) & 1:
                value |= 1 << b
            bit_cursor += 1
        out[i] = value
    return out


@dataclass
class FrameLayers:
    """一次层读取的结果。缺帧行为 False，其重建向量为零行、编号为 0。"""

    codes: np.ndarray          # (帧数, 层数) int64
    valid: np.ndarray          # (帧数,) bool
    reconstruction: np.ndarray # (帧数, 维度) float32
    version: int
    layers: int


class BitstreamWriter:
    """逐块累积位流，最后一次 :meth:`build` 产出完整字节。"""

    def __init__(self, segment_id: str, group: str, codebook_version: int,
                 n_frames: int, block_rows: int, code_bits,
                 layer_indices=None):
        self.segment_id = segment_id
        self.group = group
        self.codebook_version = int(codebook_version)
        self.n_frames = int(n_frames)
        self.block_rows = int(block_rows)
        self.code_bits = [int(b) for b in code_bits]
        # 每个流层对应快照中的码本层号；缺省为连续前 N 层。
        if layer_indices is None:
            layer_indices = list(range(len(self.code_bits)))
        self.layer_indices = [int(i) for i in layer_indices]
        self._payload = bytearray()
        self._blocks = []

    def add_block(self, valid_mask: np.ndarray, layer_indices) -> None:
        valid_mask = np.asarray(valid_mask, dtype=bool)
        rows = len(valid_mask)
        if len(layer_indices) != len(self.code_bits):
            raise ValueError("该块写入的层数与文件头声明不一致")
        valid_count = int(valid_mask.sum())
        for layer, indices in enumerate(layer_indices):
            indices = np.asarray(indices)
            if indices.shape != (valid_count,):
                raise ValueError(
                    f"第 {layer} 层编码数 {indices.shape} 与有效帧数 "
                    f"{valid_count} 不吻合"
                )
        bitmap = pack_bitmap(valid_mask)
        regions = []
        base = len(self._payload)
        self._payload.extend(bitmap)
        regions.append({
            "layer": -1,
            # 偏移先按负载起点记录，build() 时统一加上头长度。
            "offset": base,
            "length": len(bitmap),
            "kind": "bitmap",
        })
        for layer, indices in enumerate(layer_indices):
            blob = pack_codes(indices, self.code_bits[layer])
            start = len(self._payload)
            self._payload.extend(blob)
            regions.append({
                "layer": layer,
                "offset": start,
                "length": len(blob),
                "kind": "codes",
            })
        self._blocks.append({"rows": rows, "regions": regions})

    def build(self) -> bytes:
        import json

        # region 偏移保持为相对负载起点的字节，与 JSON 头长度解耦，
        # 因此平移与序列化顺序不会相互影响。
        header_core = {
            "magic": _MAGIC.decode("ascii"),
            "segment_id": self.segment_id,
            "group": self.group,
            "codebook_version": self.codebook_version,
            "frames": self.n_frames,
            "block_rows": self.block_rows,
            "layers": len(self.code_bits),
            "layer_indices": list(self.layer_indices),
            "code_bits": list(self.code_bits),
            "blocks": self._blocks,
        }
        json_bytes = json.dumps(header_core, ensure_ascii=False).encode("utf-8")
        header_blob = _MAGIC + len(json_bytes).to_bytes(4, "big") + json_bytes
        return len(header_blob).to_bytes(8, "big") + header_blob + \
            bytes(self._payload)


class BitstreamReader:
    """按头偏移随机访问任一块、任一层；不预读高层字节。"""

    def __init__(self, data: bytes):
        self.data = bytes(data)
        if len(self.data) < 8:
            raise TruncatedLayer("", -1, -1, len(self.data), 8)
        header_len = int.from_bytes(self.data[:8], "big")
        if len(self.data) < 8 + header_len:
            raise TruncatedLayer("", -1, -1, len(self.data), 8 + header_len)
        header_blob = self.data[8:8 + header_len]
        if not header_blob.startswith(_MAGIC):
            raise ValueError("不是本约定的残差量化位流")
        import json
        self.header = json.loads(header_blob[8:].decode("utf-8"))
        self._payload_start = 8 + header_len

    @property
    def segment_id(self) -> str:
        return self.header["segment_id"]

    @property
    def group(self) -> str:
        return self.header["group"]

    @property
    def declared_version(self) -> int:
        return int(self.header["codebook_version"])

    @property
    def declared_layers(self) -> int:
        return int(self.header["layers"])

    @property
    def n_blocks(self) -> int:
        return len(self.header["blocks"])

    def _region(self, block: int, layer: int, kind: str):
        entry = self.header["blocks"][block]
        for region in entry["regions"]:
            if region["kind"] == kind and region["layer"] == layer:
                return entry, region
        raise KeyError((block, layer, kind))

    def _read_region(self, block: int, layer: int, kind: str) -> bytes:
        entry, region = self._region(block, layer, kind)
        # 头中的偏移相对负载起点，读取时换算为文件绝对坐标。
        start = self._payload_start + region["offset"]
        length = region["length"]
        if len(self.data) < start + length:
            if kind == "codes":
                raise TruncatedLayer(
                    self.segment_id, layer, block,
                    max(0, len(self.data) - start), length,
                )
            raise TruncatedLayer(self.segment_id, layer, block,
                                 len(self.data), start + length)
        return self.data[start:start + length]

    def read_bitmap(self, block: int) -> np.ndarray:
        entry = self.header["blocks"][block]
        blob = self._read_region(block, -1, "bitmap")
        return unpack_bitmap(blob, entry["rows"])

    def read_block_codes(self, block: int, layer: int) -> np.ndarray:
        """返回该块该层全部有效帧的码字编号（已排除缺帧位置）。"""
        if layer >= self.declared_layers:
            raise LayerNotDeclared(
                self.segment_id, layer + 1, self.declared_layers
            )
        bits = self.header["code_bits"][layer]
        valid = self.read_bitmap(block)
        blob = self._read_region(block, layer, "codes")
        return unpack_codes(blob, int(valid.sum()), bits)

    def read_frame_code(self, block: int, row_in_block: int,
                        layer: int) -> tuple[bool, int]:
        """随机访问单帧单层：缺帧返回 ``(False, 0)``。"""
        if layer >= self.declared_layers:
            raise LayerNotDeclared(
                self.segment_id, layer + 1, self.declared_layers
            )
        valid = self.read_bitmap(block)
        if not bool(valid[row_in_block]):
            return False, 0
        codes = self.read_block_codes(block, layer)
        valid_index = int(valid[:row_in_block].sum())
        return True, int(codes[valid_index])

    def read_layers(self, up_to_layer: int, books_by_group,
                    expected_version: int | None = None) -> FrameLayers:
        """读取前 ``up_to_layer`` 层并重建。

        ``books_by_group[group]`` 为该组各层码本数组，版本由
        ``expected_version`` 校验；查不到或不匹配抛
        :class:`UnknownCodebookVersion`。未读取的高层完全不参与。
        """
        if up_to_layer > self.declared_layers:
            raise LayerNotDeclared(
                self.segment_id, up_to_layer, self.declared_layers
            )
        if expected_version is not None and \
                expected_version != self.declared_version:
            raise UnknownCodebookVersion(
                self.segment_id, self.declared_version
            )
        group = self.group
        if group not in books_by_group:
            raise UnknownCodebookVersion(
                self.segment_id, self.declared_version
            )
        books = books_by_group[group]
        needed_snapshot_layers = self.header["layer_indices"][:up_to_layer]
        if max(needed_snapshot_layers, default=0) >= len(books):
            raise UnknownCodebookVersion(
                self.segment_id, self.declared_version
            )
        n_frames = self.header["frames"]
        dim = np.asarray(books[0]).shape[1]
        codes_out = np.zeros((n_frames, up_to_layer), dtype=np.int64)
        valid_out = np.zeros(n_frames, dtype=bool)
        recon = np.zeros((n_frames, dim), dtype=np.float32)
        cursor = 0
        for block in range(self.n_blocks):
            valid = self.read_bitmap(block)
            rows = len(valid)
            valid_out[cursor:cursor + rows] = valid
            for layer in range(up_to_layer):
                blob = self._read_region(block, layer, "codes")
                indices = unpack_codes(
                    blob, int(valid.sum()), self.header["code_bits"][layer]
                )
                snapshot_layer = self.header["layer_indices"][layer]
                book = np.asarray(books[snapshot_layer], dtype=np.float32)
                chosen = book[indices]
                recon_view = recon[cursor:cursor + rows]
                recon_view[valid] += chosen
                codes_view = codes_out[cursor:cursor + rows, layer]
                codes_view[valid] = indices
            cursor += rows
        return FrameLayers(
            codes=codes_out,
            valid=valid_out,
            reconstruction=recon,
            version=self.declared_version,
            layers=up_to_layer,
        )
