import torch
import torch.nn.functional as F


def apply_delay_pattern(codes: torch.Tensor, mask_token: int):
    codes = F.pad(codes, (0, codes.shape[1]), value=mask_token)
    return torch.stack([codes[:, k].roll(k + 1) for k in range(codes.shape[1])], dim=1)


def revert_delay_pattern(codes: torch.Tensor):
    _, n_q, seq_len = codes.shape
    return torch.stack([codes[:, k, k + 1 : seq_len - n_q + k + 1] for k in range(n_q)], dim=1)