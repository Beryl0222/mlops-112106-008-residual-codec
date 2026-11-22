"""多级残差向量量化的分块编码。

约定（见 exchange.md）：

* 层间残差 = 编码前向量 − 此前已选码字之和；
* 每层选平方欧氏距离最小的有效码，平局取较小编号（``argmin`` 天然满足）；
* ``mask`` 为假的帧不参与距离，也不参与后续码本更新，在输出中不占位；
* 已标记的零向量是正常候选，缺帧与码字 0 由位流位图区分；
* 全程分块处理，距离矩阵最大只到 ``块行数 × 码字数``。
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BlockEncoding:
    """单块单层的编码事实，训练侧直接消费。"""

    valid_mask: np.ndarray
    # 每层有效帧的码字编号，长度等于 valid_mask 中 True 的个数。
    layer_indices: list
    # 每层进入该层前的有效帧残差（即该层的输入残差）。
    layer_residuals: list
    # 每层有效帧的平方量化误差 ||r − 选中码字||^2。
    layer_errors: list
    # 块内重建向量，仅对有效帧有意义。
    reconstruction: np.ndarray


def nearest_codes(residuals: np.ndarray, book: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (码字编号, 平方距离)。平局由 ``argmin`` 取较小编号。

    距离按 float64 累加，避免大码本下的舍入改变平局结果。
    """
    residuals = np.asarray(residuals, dtype=np.float64)
    book = np.asarray(book, dtype=np.float64)
    # ||r-c||^2 = ||r||^2 + ||c||^2 - 2 r·c，分块点乘不构造全片矩阵。
    r_norm = np.einsum("ij,ij->i", residuals, residuals)
    c_norm = np.einsum("ij,ij->i", book, book)
    dots = residuals @ book.T
    distances = r_norm[:, None] + c_norm[None, :] - 2.0 * dots
    indices = np.argmin(distances, axis=1)
    best = distances[np.arange(len(indices)), indices]
    return indices.astype(np.int64), best


def iter_block_ranges(n_frames: int, block_rows: int):
    for start in range(0, n_frames, block_rows):
        yield start, min(n_frames, start + block_rows)


def encode_block(frames: np.ndarray, valid_mask: np.ndarray,
                 books) -> BlockEncoding:
    """对一个块执行完整 RVQ 编码。

    ``books`` 为该组各层码本；调用方按获准交付的层数裁剪后传入。
    缺帧位置不进入任何层的距离打分，重建向量保持零。
    """
    frames = np.asarray(frames, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    valid = valid_mask
    residual_all = frames.copy()
    recon = np.zeros_like(frames)
    layer_indices, layer_residuals, layer_errors = [], [], []
    for book in books:
        book = np.asarray(book, dtype=np.float32)
        residual_valid = residual_all[valid]
        indices, distances = nearest_codes(residual_valid, book)
        chosen = book[indices]
        # 该层输入残差与平方误差即训练事实，全部按有效帧冻结。
        layer_residuals.append(np.asarray(residual_valid, dtype=np.float32))
        layer_indices.append(indices)
        layer_errors.append(distances.astype(np.float64))
        recon[valid] += chosen
        residual_all[valid] = residual_valid - chosen
    return BlockEncoding(
        valid_mask=valid,
        layer_indices=layer_indices,
        layer_residuals=layer_residuals,
        layer_errors=layer_errors,
        reconstruction=recon,
    )
