import torch


def extract_features(sequence_tensor: torch.Tensor) -> torch.Tensor:
    """
    モーションテンソルから特徴量を抽出する基本関数。

    Parameters:
    - sequence_tensor (torch.Tensor): 形状 (1, 時間ステップ, 関節次元) のテンソル

    Returns:
    - feature_vector (torch.Tensor): 形状 (特徴次元,) の1次元テンソル
    """
    if sequence_tensor.dim() != 3:
        raise ValueError("Input tensor must be 3D: (1, T, D)")

    # 時間方向の統計量（平均・分散・最大・最小）
    mean = sequence_tensor.mean(dim=1).squeeze(0)       # (D,)
    std = sequence_tensor.std(dim=1).squeeze(0)         # (D,)
    max_ = sequence_tensor.max(dim=1).values.squeeze(0) # (D,)
    min_ = sequence_tensor.min(dim=1).values.squeeze(0) # (D,)

    # 特徴量を結合（例: D次元×4 = 204次元）
    feature_vector = torch.cat([mean, std, max_, min_], dim=0)

    return feature_vector
