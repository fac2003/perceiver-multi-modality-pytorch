<img src="./perceiver.png" width="600px"></img>

## Perceiver - Pytorch

Implementation of <a href="https://arxiv.org/abs/2103.03206">Perceiver</a>, General Perception with Iterative Attention, in Pytorch

<a href="https://www.youtube.com/watch?v=P_xeshTnPZg">Yannic Kilcher explanation!</a>

## Install

```bash
$ pip install perceiver-pytorch
```

## Usage

```python
import torch
from perceiver_pytorch import Perceiver

model = Perceiver(
    input_channels = 3,          # number of channels for each token of the input
    input_axis = 2,              # number of axis for input data (2 for images, 3 for video)
    num_freq_bands = 6,          # number of freq bands, with original value (2 * K + 1)
    max_freq = 10.,              # maximum frequency, hyperparameter depending on how fine the data is
    depth = 6,                   # depth of net
    num_latents = 256,           # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim = 512,            # latent dimension
    cross_heads = 1,             # number of heads for cross attention. paper said 1
    latent_heads = 8,            # number of heads for latent self attention, 8
    cross_dim_head = 64,
    latent_dim_head = 64,
    num_classes = 1000,          # output number of classes
    attn_dropout = 0.,
    ff_dropout = 0.,
    weight_tie_layers = False    # whether to weight tie layers (optional, as indicated in the diagram)
)

img = torch.randn(1, 224, 224, 3) # 1 imagenet image, pixelized

model(img) # (1, 1000)
```

## Experimental

I have also included a version of Perceiver that includes bottom-up (in addition to top-down) attention, using the same scheme as presented in the original <a href="https://arxiv.org/abs/1810.00825">Set Transformers</a> paper as the <a href="https://github.com/lucidrains/isab-pytorch">Induced Set Attention Block</a>.

You simply have to change the above import to

```python
from perceiver_pytorch.experimental import Perceiver
```

### Multi-modality perceiver
An attractive feature of the perceiver architecture is that it can process multiple modalities of data 
in the same batch. This is not obvious from the perceiver forward signature shown above, but a relatively
modest change can support processing video, images and audio with a single model, in one forward.
This feature is demonstrated by the MultiModalityPerceiver, contributed by Fabien Campagne.

```python
from perceiver_pytorch.multi_modality_perceiver import MultiModalityPerceiver, InputModality


image_inputs= torch.rand(size=(3, 260, 260, 3), requires_grad=True)
video_inputs= torch.rand(size=(3, 32, 260, 260, 3), requires_grad=True)
audio_inputs= torch.rand(size=(3, 44100,1), requires_grad=True)

video_modality = InputModality(
    name='video',
    input_channels=3,  # number of channels for each token of the input
    input_axis=3,  # number of axes, 3 for video)
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
)
image_modality = InputModality(
    name='image',
    input_channels=3,  # number of channels for each token of the input
    input_axis=2,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=4.,  # maximum frequency, hyperparameter depending on how fine the data is
)
audio_modality = InputModality(
    name='audio',
    input_channels=1,  # number of channels for mono audio
    input_axis=1,  # number of axes, 2 for images
    num_freq_bands=6,  # number of freq bands, with original value (2 * K + 1)
    max_freq=8.,  # maximum frequency, hyperparameter depending on how fine the data is
)
model = MultiModalityPerceiver(
    modalities=(video_modality, image_modality, audio_modality),
    depth=6,  # depth of net
    num_latents=12,
    # number of latents, or induced set points, or centroids. different papers giving it different names
    cross_dim=64,  # cross attention dimension
    latent_dim=64,  # latent dimension
    cross_heads=1,  # number of heads for cross attention. paper said 1
    latent_heads=8,  # number of heads for latent self attention, 8
    cross_dim_head=64,
    latent_dim_head=64,
    num_classes=1000,  # output number of classes
    attn_dropout=0.,
    ff_dropout=0.,
    weight_tie_layers=True
    # whether to weight tie layers (optional, as indicated in the diagram)
)
result = model({'image': image_inputs,
                'video': video_inputs,
                'audio': audio_inputs})
```
## Citations

```bibtex
@misc{jaegle2021perceiver,
    title   = {Perceiver: General Perception with Iterative Attention},
    author  = {Andrew Jaegle and Felix Gimeno and Andrew Brock and Andrew Zisserman and Oriol Vinyals and Joao Carreira},
    year    = {2021},
    eprint  = {2103.03206},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
