def _compute_exit_mask_v3(
        self,
        curr_out: Dict[str, torch.Tensor],
        prev_out: Dict[str, torch.Tensor],
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    基于当前 head 可解码信息的 early exit:
        - center: (B, N, 3)
        - size:   (B, N, 3)
        - angle:  (B, N, 2)   # 2D angle representation, e.g. [sin, cos] or equivalent
        - class:  (B, N, C)

    统一稳定性指标:
        box_delta =
            lambda_center * center_delta_rel +
            lambda_size   * size_delta_rel   +
            lambda_angle  * angle_delta

    其中:
        center_delta_rel:
            前后两轮中心变化，按平均 box size 归一化后做 L2

        size_delta_rel:
            前后两轮尺寸变化，按平均 box size 归一化后做 L2

        angle_delta:
            前后两轮 angle 2D 表示先单位化，再做向量差 L2
            不依赖具体 yaw 解码方式，直接适配当前 head 输出

    early exit 条件:
        1) box_delta 足够小
        2) 可选: score 足够高
    """
    if prev_out is None:
        return torch.zeros_like(valid_mask)

    eps = 1e-8
    min_size = getattr(self, "exit_min_size", 0.1)

    lambda_center = getattr(self, "exit_lambda_center", 1.0)
    lambda_size = getattr(self, "exit_lambda_size", 0.7)
    lambda_angle = getattr(self, "exit_lambda_angle", 0.3)

    use_score_gate = getattr(self, "exit_use_score_gate", False)
    score_threshold = getattr(self, "exit_score_threshold", 0.3)

    # -----------------------------
    # 1) center / size
    # -----------------------------
    prev_center = prev_out["center"][..., :3]
    curr_center = curr_out["center"][..., :3]

    prev_size = prev_out["size"][..., :3].abs()
    curr_size = curr_out["size"][..., :3].abs()

    avg_size = 0.5 * (prev_size + curr_size)
    avg_size = avg_size.clamp_min(min_size).clamp_min(eps)

    # center relative delta
    center_diff = (curr_center - prev_center).abs()
    center_delta_rel = (center_diff / avg_size).norm(dim=-1)  # (B, N)

    # size relative delta
    size_diff = (curr_size - prev_size).abs()
    size_delta_rel = (size_diff / avg_size).norm(dim=-1)  # (B, N)

    # -----------------------------
    # 2) angle: (B, N, 2)
    # 不直接解 yaw，直接比较归一化方向向量
    # -----------------------------
    prev_angle = prev_out.get("angle", None)
    curr_angle = curr_out.get("angle", None)

    if prev_angle is not None and curr_angle is not None:
        # 归一化到单位向量，避免幅值变化干扰角度稳定性
        prev_angle_unit = prev_angle / prev_angle.norm(dim=-1, keepdim=True).clamp_min(eps)
        curr_angle_unit = curr_angle / curr_angle.norm(dim=-1, keepdim=True).clamp_min(eps)

        # 方向差，范围大致在 [0, 2]
        angle_delta = (curr_angle_unit - prev_angle_unit).norm(dim=-1)  # (B, N)
    else:
        angle_delta = center_delta_rel.new_zeros(center_delta_rel.shape)

    # -----------------------------
    # 3) unified stability metric
    # -----------------------------
    box_delta = (
            lambda_center * center_delta_rel
            + lambda_size * size_delta_rel
            + lambda_angle * angle_delta
    )

    stable = box_delta <= self.exit_box_threshold

    # -----------------------------
    # 4) optional score gate
    # 只做弱约束，防止明显低质量 query 过早退出
    # -----------------------------
    if use_score_gate:
        confident = scores >= score_threshold
        stable = stable & confident

    return valid_mask & stable


def _compute_exit_mask_v1(self, curr_out: Dict[str, torch.Tensor], prev_out: Dict[str, torch.Tensor],
                          scores: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """ 计算哪些 query 可以 early exit。 early exit
        条件： 1. 当前 query 有效 2. 分类分数高于阈值 3.
        当前轮与上一轮相比，框已经足够稳定 - center 变化小 - size 变化小 - angle 变化小
        返回： exit_mask: (B, N)，True 表示该 query 提前退出，不再参加后续 refinement """

    if prev_out is None:
        return torch.zeros_like(valid_mask)
    # 中心变化：欧氏距离
    center_delta = (curr_out['center'][..., :3] - prev_out['center'][..., :3]).norm(dim=-1)
    # 尺寸变化：平均绝对差
    size_delta = torch.zeros_like(center_delta)
    if 'size' in curr_out and 'size' in prev_out:
        size_delta = (curr_out['size'] - prev_out['size']).abs().mean(dim=-1)
    # 朝向变化：平均绝对差
    angle_delta = torch.zeros_like(center_delta)
    if 'angle' in curr_out and 'angle' in prev_out:
        angle_delta = (curr_out['angle'] - prev_out['angle']).abs().mean(dim=-1)
    stable = ((center_delta <= self.exit_center_threshold) & (size_delta <= self.exit_size_threshold)
              & (angle_delta <= self.exit_angle_threshold))
    confident = scores >= self.exit_score_threshold
    return valid_mask & stable & confident
    # return valid_mask & stable


def _compute_exit_mask_v2(self,
                       curr_out: Dict[str, torch.Tensor],
                       prev_out: Dict[str, torch.Tensor],
                       scores: torch.Tensor,
                       valid_mask: torch.Tensor) -> torch.Tensor:
    """
    基于 box 整体相对变化做 early exit。

    设计目标：
        1. 不再分别对 center / size / angle 设多个阈值
            2. 只保留一个统一的 box 稳定性指标 box_delta
            3. 不再依赖 score，也不使用 angle
            4. 只要 box 已经足够稳定，就允许该 query 提前退出

        box_delta 定义：
            box_delta = center_delta_rel + lambda_size * size_delta_rel

        其中：
            center_delta_rel:
                当前轮和上一轮中心差，按 box 各维尺度逐维归一化后，再做 L2 聚合
            size_delta_rel:
                当前轮和上一轮尺寸差，按前后两轮平均尺度归一化后，再做 L2 聚合

        返回：
            exit_mask: (B, N)，True 表示该 query 提前退出
        """
    if prev_out is None:
        return torch.zeros_like(valid_mask)

    eps = 1e-8
    min_size = 0.1
    # 尺寸变化的占比
    center_dims = 3

    # -----------------------------
    # 取前后两轮的 center / size
    # -----------------------------
    prev_center = prev_out["center"][..., :center_dims]
    curr_center = curr_out["center"][..., :center_dims]

    prev_size = prev_out["size"][..., :center_dims].abs()
    curr_size = curr_out["size"][..., :center_dims].abs()

    # -----------------------------
    # 用前后两轮尺寸均值作为归一化尺度
    # 这样更对称，也更平滑
    # -----------------------------
    scale = 0.5 * (prev_size + curr_size)
    scale = scale.clamp_min(min_size).clamp_min(eps)

    # -----------------------------
    # 1) center 相对变化
    # 逐维按 box 尺度归一化，再做 L2 聚合
    # -----------------------------
    center_diff = curr_center - prev_center
    center_rel = center_diff.abs() / scale
    center_delta_rel = center_rel.norm(dim=-1)  # shape: (B, N)

    # -----------------------------
    # 3) 单一 box 稳定性指标
    # -----------------------------
    box_delta = center_delta_rel

    # -----------------------------
    # 4) 单一阈值判断 early exit
    # -----------------------------
    stable = box_delta <= self.exit_box_threshold

    return valid_mask & stable

# 基于IoU
def _compute_exit_mask_v4(
        self,
        curr_out: Dict[str, torch.Tensor],
        prev_out: Dict[str, torch.Tensor],
        scores: torch.Tensor,
        valid_mask: torch.Tensor,
) -> torch.Tensor:
    if prev_out is None:
        return torch.zeros_like(valid_mask)

    curr_center = curr_out["center"]
    curr_size = curr_out["size"]
    prev_center = prev_out["center"]
    prev_size = prev_out["size"]

    eps = 1e-6

    curr_half = curr_size * 0.5
    prev_half = prev_size * 0.5

    curr_min = curr_center - curr_half
    curr_max = curr_center + curr_half
    prev_min = prev_center - prev_half
    prev_max = prev_center + prev_half

    inter_min = torch.maximum(curr_min, prev_min)
    inter_max = torch.minimum(curr_max, prev_max)
    inter_size = (inter_max - inter_min).clamp(min=0.0)

    inter_vol = inter_size[..., 0] * inter_size[..., 1] * inter_size[..., 2]
    curr_vol = (curr_size[..., 0] * curr_size[..., 1] * curr_size[..., 2]).clamp(min=0.0)
    prev_vol = (prev_size[..., 0] * prev_size[..., 1] * prev_size[..., 2]).clamp(min=0.0)
    union_vol = curr_vol + prev_vol - inter_vol

    iou3d = inter_vol / union_vol.clamp(min=eps)
    box_delta = 1.0 - iou3d

    stable = box_delta <= self.exit_box_threshold

    if hasattr(self, "exit_center_threshold") and self.exit_center_threshold is not None:
        center_shift = torch.norm(curr_center - prev_center, dim=-1)
        stable = stable & (center_shift <= self.exit_center_threshold)

    if scores is not None and hasattr(self, "exit_score_threshold") and self.exit_score_threshold is not None:
        if scores.ndim == 3:
            score_values = scores.max(dim=-1).values
        else:
            score_values = scores
        stable = stable & (score_values >= self.exit_score_threshold)

    return valid_mask & stable
