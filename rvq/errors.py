"""残差量化码流各环节的显式异常。

消费端必须能区分三种读取结果：声明里根本没有该高层、层内容字节不足、
码本版本在留存档案中查不到；其余异常对应训练与容量约束。
"""


class RvqError(Exception):
    """本包所有异常的基类。"""


class LayerNotDeclared(RvqError):
    """请求的层数超过该片段实际写入（获准交付）的层边界。"""

    def __init__(self, segment_id: str, requested: int, declared: int):
        self.segment_id = segment_id
        self.requested = requested
        self.declared = declared
        super().__init__(
            f"片段 {segment_id!r} 仅声明 {declared} 层，请求读取第 {requested} 层"
        )


class TruncatedLayer(RvqError):
    """位流在某个明确的层边界之内缺字节，低层仍可能完整可读。"""

    def __init__(self, segment_id: str, layer: int, block: int,
                 have_bytes: int, need_bytes: int):
        self.segment_id = segment_id
        self.layer = layer
        self.block = block
        self.have_bytes = have_bytes
        self.need_bytes = need_bytes
        super().__init__(
            f"片段 {segment_id!r} 第 {block} 块第 {layer} 层内容不足："
            f"需要 {need_bytes} 字节，实际 {have_bytes} 字节"
        )


class UnknownCodebookVersion(RvqError):
    """位流头记录的码本整版在当前留存档案中查不到。"""

    def __init__(self, segment_id: str, version: int):
        self.segment_id = segment_id
        self.version = version
        super().__init__(
            f"片段 {segment_id!r} 引用的码本版本 {version} 不在留存档案中"
        )


class AlreadyCommitted(RvqError):
    """同一提交重放：该层（或整片段）已学习过，不能重复计入。"""

    def __init__(self, segment_id: str, layers):
        self.segment_id = segment_id
        self.layers = tuple(layers)
        super().__init__(
            f"片段 {segment_id!r} 的层 {tuple(self.layers)} 已提交，重放不重复学习"
        )


class RetentionExceeded(RvqError):
    """留存容量不足，最旧版本仍被未消费片段引用，不能剔除。"""

    def __init__(self, version: int, max_versions: int, references):
        self.version = version
        self.max_versions = max_versions
        # references: list[dict]，每项描述一个仍引用旧版的未消费片段
        self.references = list(references)
        super().__init__(
            f"码本留存容量 {max_versions} 已满；版本 {version} 仍被 "
            f"{len(self.references)} 个未消费片段引用"
        )


class MemoryBudgetExceeded(RvqError):
    """待交付状态与计算暂存超过产品内存预算。"""

    def __init__(self, need_bytes: int, budget_bytes: int, detail: str = ""):
        self.need_bytes = need_bytes
        self.budget_bytes = budget_bytes
        self.detail = detail
        super().__init__(
            f"需要 {need_bytes} 字节，超出预算 {budget_bytes} 字节。{detail}"
        )


class InconsistentSubmission(RvqError):
    """编码/提交输入与码本或片段约束不一致。"""
