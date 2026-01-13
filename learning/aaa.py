
# ---- 计算目标尺寸（与 Resize.get_size 等价）----
def constrain_to_multiple_of(x, multiple, min_val=None, max_val=None):
    # numpy 是 round，这里用 torch 实现近似
    # 输入可能是标量，直接用 python 计算
    import math
    y = int(round(x / multiple) * multiple)
    if max_val is not None and y > max_val:
        y = int(math.floor(x / multiple) * multiple)
    if min_val is not None and y < min_val:
        y = int(math.ceil(x / multiple) * multiple)
    return y

def preprocess_isaac_rgb(
        imgs_b3hw: torch.Tensor,
        input_size: int = 518,
        keep_aspect_ratio: bool = True,
        ensure_multiple_of: int = 14,
        resize_method: str = "lower_bound",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        device: str = None,
):
    """
    等价于：
      Resize(width=input_size, height=input_size, keep_aspect_ratio=True,
             ensure_multiple_of=14, resize_method='lower_bound', cv2.INTER_CUBIC)
      -> NormalizeImage(mean, std)
      -> PrepareForNet()  (NCHW, float32)

    Args:
        imgs_b3hw: 形状 (B, 3, H, W) 的 torch.Tensor，RGB。范围可以是[0,1]或[0,255]。
        input_size: 目标输入边长（与原代码一致，518）
        keep_aspect_ratio, ensure_multiple_of, resize_method: 同你的 Resize 配置
        mean, std: 归一化参数
        device: 若为 None，则保持输入所在 device；否则 to(device)
    Returns:
        proc: 预处理后的 (B, 3, H_new, W_new) 张量（float32, 已归一化）
        orig_hw: 原始 (H, W)，tuple
        new_hw:  新尺寸 (H_new, W_new)，tuple
    """
    assert imgs_b3hw.ndim == 4 and imgs_b3hw.shape[1] == 3, "Expect (B, 3, H, W) tensor"
    B, C, H, W = imgs_b3hw.shape
    orig_hw = (H, W)

    if device is None:
        device = imgs_b3hw.device
    imgs = imgs_b3hw.to(device=device, dtype=torch.float32)

    # 若是 0~255，转 0~1
    if imgs.max() > 1.0:
        imgs = imgs / 255.0

    # 根据 keep_aspect_ratio & resize_method 计算 scale
    scale_h = input_size / H
    scale_w = input_size / W

    if keep_aspect_ratio:
        if resize_method == "lower_bound":
            # 输出至少不小于 input_size
            scale = max(scale_h, scale_w)
        elif resize_method == "upper_bound":
            scale = min(scale_h, scale_w)
        elif resize_method == "minimal":
            # 动得最少
            scale = scale_w if abs(1 - scale_w) < abs(1 - scale_h) else scale_h
        else:
            raise ValueError(f"Unknown resize_method: {resize_method}")
        new_h = int(scale * H)
        new_w = int(scale * W)
    else:
        new_h = int(scale_h * H)
        new_w = int(scale_w * W)

    if resize_method == "lower_bound":
        new_h = constrain_to_multiple_of(new_h, ensure_multiple_of, min_val=input_size)
        new_w = constrain_to_multiple_of(new_w, ensure_multiple_of, min_val=input_size)
    elif resize_method == "upper_bound":
        new_h = constrain_to_multiple_of(new_h, ensure_multiple_of, max_val=input_size)
        new_w = constrain_to_multiple_of(new_w, ensure_multiple_of, max_val=input_size)
    elif resize_method == "minimal":
        new_h = constrain_to_multiple_of(new_h, ensure_multiple_of)
        new_w = constrain_to_multiple_of(new_w, ensure_multiple_of)
    else:
        raise ValueError(f"Unknown resize_method: {resize_method}")

    new_hw = (new_h, new_w)

    # ---- resize（cv2.INTER_CUBIC ~= bicubic）----
    imgs = F.interpolate(imgs, size=new_hw, mode="bicubic", align_corners=False)

    # ---- NormalizeImage ----
    mean_t = torch.tensor(mean, device=device, dtype=imgs.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=imgs.dtype).view(1, 3, 1, 1)
    imgs = (imgs - mean_t) / std_t  # 函数 PrepareForNet 本身已是 (C,H,W)，这里不需要再转置

    # imgs 已经是 float32, NCHW, contiguous
    return imgs.contiguous(), orig_hw, new_hw