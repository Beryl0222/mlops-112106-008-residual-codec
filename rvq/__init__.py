"""离线字幕设备的多级残差向量量化与码流工具包。"""

from .archive import EncodedSegment, QuantizationArchive
from .bitstream import (
    BitstreamReader,
    BitstreamWriter,
    FrameLayers,
    bits_for_codes,
    pack_bitmap,
    pack_codes,
    unpack_bitmap,
    unpack_codes,
)
from .budget import MemoryBudget
from .errors import (
    AlreadyCommitted,
    InconsistentSubmission,
    LayerNotDeclared,
    MemoryBudgetExceeded,
    RetentionExceeded,
    RvqError,
    TruncatedLayer,
    UnknownCodebookVersion,
)
from .quantize import BlockEncoding, encode_block, nearest_codes
from .training import CodebookTrainer, LayerStats
from .versions import VersionStore

__all__ = [
    "QuantizationArchive", "EncodedSegment",
    "BitstreamReader", "BitstreamWriter", "FrameLayers",
    "bits_for_codes", "pack_bitmap", "unpack_bitmap",
    "pack_codes", "unpack_codes",
    "MemoryBudget", "CodebookTrainer", "LayerStats", "VersionStore",
    "BlockEncoding", "encode_block", "nearest_codes",
    "RvqError", "LayerNotDeclared", "TruncatedLayer",
    "UnknownCodebookVersion", "AlreadyCommitted", "RetentionExceeded",
    "MemoryBudgetExceeded", "InconsistentSubmission",
]
