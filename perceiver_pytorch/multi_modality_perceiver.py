from dataclasses import dataclass
from typing import Iterable, Dict

import torch
from einops import rearrange, repeat
from torch import Tensor
from torch import nn

from perceiver_pytorch.perceiver_pytorch import PreNorm, Attention, FeedForward, cache_fn, fourier_encode


@dataclass
class InputModality:
    name: str
    input_channels: int
    input_axis: int
    num_freq_bands: int
    max_freq: float
    freq_base: int = 2

    @property
    def input_dim(self) -> int:
        # Calculate the dimension of this modality.
        input_dim = self.input_axis * ((self.num_freq_bands * 2) + 1) + self.input_channels
        return input_dim


def modality_encoding(batch_size: int, axes, modality_index: int, num_modalities: int) -> Tensor:
    """
    Return one-hot encoding of modality given num_modalities, batch size and axes.
    The result need to be compatible with the modality data for concatenation.
    :param modality_index:
    :param num_modalities:
    :return:
    """
    one_hot = torch.eye(num_modalities, num_modalities)[modality_index]
    to_expand = [batch_size]
    one_hot = one_hot.unsqueeze(0)
    for i, axis in enumerate(axes):
        one_hot = one_hot.unsqueeze(0)
        to_expand.append(axis)
    to_expand.append(num_modalities)

    one_hot = one_hot.expand(to_expand)
    return one_hot


# An implementation of Perceiver that can accept multiple data modalities in the same forward.
class MultiModalityPerceiver(nn.Module):
    def __init__(
            self,
            *,
            modalities: Iterable[InputModality],
            depth,
            num_latents=512,
            cross_dim=512,
            latent_dim=512,
            cross_heads=1,
            latent_heads=8,
            cross_dim_head=64,
            latent_dim_head=64,
            num_classes=1000,
            attn_dropout=0.,
            ff_dropout=0.,
            weight_tie_layers=False
    ):
        super().__init__()
        self.modalities = {modality.name: modality for modality in modalities}
        # we encode modality with one hot encoding, so need one dim per modality:
        modality_encoding_dim = sum([1 for _ in modalities])
        # input_dim is the maximum dimension over all input modalities:
        input_dim = max(modality.input_dim for modality in modalities) + modality_encoding_dim
        self.max_modality_dim = input_dim
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim,
                                         Attention(latent_dim, input_dim, heads=cross_heads, dim_head=cross_dim_head,
                                                   dropout=attn_dropout), context_dim=input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim,
                                          Attention(latent_dim, heads=latent_heads, dim_head=latent_dim_head,
                                                    dropout=attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout=ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (
            get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self.to_logits = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        )

    def forward(self, multi_modality_data: Dict[str, Tensor], mask=None):
        """

        :param data: a dictionary where keys are modality names and Tensor contain a batch
        of modality input data.
        :param mask:
        :return:
        """
        batch_sizes = set()
        num_modalities = len(multi_modality_data)
        linearized_data = []
        for modality_index, modality_name in enumerate(sorted(multi_modality_data.keys())):
            data = multi_modality_data[modality_name]
            modality = self.modalities[modality_name]
            b, *axis, _, device = *data.shape, data.device
            assert len(axis) == modality.input_axis, 'input data must have the right number of axis'
            batch_sizes.add(b)
            assert len(batch_sizes) == 1, "batch size must be the same across all modalities"
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim=-1)
            enc_pos = fourier_encode(pos,
                                     modality.max_freq, modality.num_freq_bands, modality.freq_base)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b=b)

            # Figure out padding for this modality, given max dimension across all modalities:
            padding_size = self.max_modality_dim - modality.input_dim - num_modalities
            print(f"padding_size={padding_size} for modality={modality_name}")
            padding = torch.zeros(size=data.size()[0:-1] + (padding_size,))
            # concat to channels of data and flatten axis
            modality_encodings = modality_encoding(b, axis, modality_index, num_modalities)

            to_concat = (data, padding, enc_pos, modality_encodings)

            data = torch.cat(to_concat, dim=-1)
            data = rearrange(data, 'b ... d -> b (...) d')
            linearized_data.append(data)
        b = batch_sizes.pop()
        x = repeat(self.latents, 'n d -> b n d', b=b)

        # Concatenate all the modalities:
        data = torch.cat(linearized_data, dim=1)

        for cross_attn, cross_ff, latent_attn, latent_ff in self.layers:
            x = cross_attn(x, context=data, mask=mask) + x
        x = cross_ff(x) + x
        x = latent_attn(x) + x
        x = latent_ff(x) + x

        x = x.mean(dim=-2)
        return self.to_logits(x)
