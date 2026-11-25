"""Common causal temporal encoder, origin context and separate future slots."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, width, kernel, dilation):
        super().__init__()
        self.left = dilation*(kernel-1)
        self.conv = nn.Conv1d(width, width, kernel, dilation=dilation)
        self.norm = nn.LayerNorm(width)

    def forward(self, x):
        y = self.conv(F.pad(x, (self.left, 0)))
        return x + F.gelu(self.norm(y.transpose(1, 2))).transpose(1, 2)


class ForecastNet(nn.Module):
    def __init__(self, features, cfg, sigma, outputs=2):
        super().__init__()
        width = cfg['model']['hidden_dim']
        self.features = features
        self.input = nn.Linear(features, width)
        self.temporal = nn.Sequential(*[
            CausalBlock(width, cfg['model']['kernel_size'], d)
            for d in cfg['model']['temporal_dilations']])
        self.context = nn.Sequential(nn.Linear(2*width, width), nn.GELU(), nn.LayerNorm(width))
        self.future_encoder = nn.Sequential(
            nn.Linear(3*(features+1)+4, width), nn.GELU(), nn.LayerNorm(width))
        self.fusion = nn.Sequential(nn.Linear(3*width, width), nn.GELU(), nn.LayerNorm(width))
        self.output = nn.Linear(width, outputs)
        scales = [sigma, 1.] if outputs == 2 else [sigma, sigma, 1., 1.]
        self.register_buffer('output_scale', torch.tensor(scales, dtype=torch.float32))

    def encode(self, x, future=None, k=0):
        h = self.temporal(self.input(x).transpose(1, 2))[:, :, -1]
        pooled = torch.cat([h.mean(0), h.std(0, unbiased=False)])
        context = self.context(pooled).expand(len(x), -1)
        if future is None:
            future = x.new_zeros((len(x), 3, self.features+1))
        marker = x.new_zeros((len(x), 4))
        marker[:, k] = 1
        f = self.future_encoder(torch.cat([future.flatten(1), marker], 1))
        return self.fusion(torch.cat([h, context, f], 1))

    def forward(self, x, future=None, k=0):
        return self.output(self.encode(x, future, k))*self.output_scale


def mse(prediction, target, mask):
    if not bool(mask.any()):
        return None
    return (prediction[mask]-target[mask]).square().mean()


def pair_mse(prediction, target, mask, weights=None):
    ids = torch.nonzero(mask, as_tuple=False).flatten()
    if len(ids) < 2:
        return prediction.sum()*0
    error = prediction[ids]-target[ids]
    if weights is None:
        return 2*error.sub(error.mean()).square().sum()/(len(ids)-1)
    w = weights[ids][:, ids]
    return (w*(error[:, None]-error[None, :]).square()).sum()/w.sum().clamp_min(1e-12)


def student_loss(output, y, v, mask, rank_mask, variance, cfg, weights=None, listnet=False):
    point = mse(output[:, 0], y, mask)
    if point is None:
        return None
    pair = pair_mse(output[:, 0], y, mask, weights)/variance
    rank = output.sum()*0
    if bool(rank_mask.any()):
        a, b = output[rank_mask, 1], v[rank_mask]
        rank = -(b.softmax(0)*a.log_softmax(0)).sum() if listnet else (a-b).square().mean()
    return point/variance + cfg['training']['pair_weight']*pair + cfg['training']['rank_weight']*rank
