import torch
import torch.nn as nn
import math
import warnings


def get_pad_mask(seq, pad_idx=0):
    if seq.dim() != 2:
        raise ValueError("<seq> has to be a 2-dimensional tensor!")
    if not isinstance(pad_idx, int):
        raise TypeError("<pad_index> has to be an int!")

    return (seq != pad_idx).unsqueeze(1) # equivalent (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq, diagonal=1):
    if seq.dim() < 2:
        raise ValueError("<seq> has to be at least a 2-dimensional tensor!")

    seq_len = seq.size(1)
    mask = (1 - torch.triu(torch.ones((1, seq_len, seq_len), device=seq.device), diagonal=diagonal)).bool()
    return mask


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def traj_affordance_dist(hand_traj, contact_point, future_valid=None, invalid_value=9):
    batch_size = contact_point.shape[0]
    expand_size = int(hand_traj.shape[0] / batch_size)
    contact_point = contact_point.unsqueeze(dim=1).expand(-1, expand_size, 2).reshape(-1, 2)  # (B * 2 * Tf, 2)
    dist = torch.sum((hand_traj - contact_point) ** 2, dim=1).reshape(batch_size, -1)  # (B, 2 * Tf)
    if future_valid is None:
        sorted_dist, sorted_idx = torch.sort(dist, dim=-1, descending=False)  # from small to high
        return sorted_dist[:, 0]  # (B, )
    else:
        dist = dist.reshape(batch_size, 2, -1)  # (B, 2, Tf)
        future_valid = future_valid > 0
        future_invalid = ~future_valid[:, :, None].expand(dist.shape)
        dist[future_invalid] = invalid_value  # set invalid dist to be very large
        sorted_dist, sorted_idx = torch.sort(dist, dim=-1, descending=False)  # from small to high
        selected_dist = sorted_dist[:, :, 0]  # (B, 2)
        selected_dist, selected_idx = selected_dist.min(dim=1)  # selected_dist, selected_idx (B, )
        valid = torch.gather(future_valid, dim=1, index=selected_idx.unsqueeze(dim=1)).squeeze(dim=1)
        selected_dist = selected_dist * valid

    return selected_dist