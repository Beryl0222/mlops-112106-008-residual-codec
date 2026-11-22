"""内存预算与分块尺寸推导。

``budget_bytes`` 按交换约定只计入跨片段的待交付状态和计算暂存，
不含一次输入块、最终归档字节以及已生效码本。长片段必须分块搜索，
任何环节都不得构造 全片帧数 × 码字数 的距离矩阵。
"""

from dataclasses import dataclass

import numpy as np

from .errors import MemoryBudgetExceeded

# 单块行数的工程上下限：下限保证进度，上限约束单次 matmul 规模。
MIN_BLOCK_ROWS = 32
MAX_BLOCK_ROWS = 4096
# 每行除距离打分外的杂项暂存系数（残差、位图、索引等），按字节估算。
PER_ROW_OVERHEAD = 64


@dataclass
class MemoryBudget:
    """跟踪待交付状态与计算暂存的字节占用。"""

    total_bytes: int
    used_bytes: int = 0

    @property
    def available(self) -> int:
        return max(0, self.total_bytes - self.used_bytes)

    def reserve(self, nbytes: int, detail: str = "") -> None:
        nbytes = int(nbytes)
        if self.used_bytes + nbytes > self.total_bytes:
            raise MemoryBudgetExceeded(
                self.used_bytes + nbytes, self.total_bytes, detail
            )
        self.used_bytes += nbytes

    def release(self, nbytes: int) -> None:
        self.used_bytes = max(0, self.used_bytes - int(nbytes))

    def fit_block_rows(self, n_dims: int, max_codes: int,
                       cap: int = MAX_BLOCK_ROWS) -> int:
        """按当前可用预算推导单块行数。

        一行的主要暂存是 ``m×K`` 的距离打分（float32）与 ``m×d`` 残差，
        因此 ``4 * (K + d) + 杂项`` 即每行字节数。
        """
        per_row = 4 * (max_codes + n_dims) + PER_ROW_OVERHEAD
        rows = self.available // max(1, per_row)
        rows = max(MIN_BLOCK_ROWS, min(cap, rows))
        return int(rows)


def numpy_bytes(arrays) -> int:
    """统计一组 NumPy 数组的实际占用字节。"""
    total = 0
    for array in arrays:
        total += int(np.asarray(array).nbytes)
    return total
