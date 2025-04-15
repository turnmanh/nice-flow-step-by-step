import pytest 
import torch

from src.flow.networks import (
    ConcatenatedELU,
    GatedConvolution,
    GatedConvolution2DNetwork,
    LayerNormChannels,
)


def test_gated_conv_network_shape() -> None:
    """Test whether the gated conv. network produces the expected output shape."""
    c_in, c_hidden, c_out = 3, 6, 6
    hight, width = 28, 28
    input_tensor = torch.randn(1, c_in, hight, width)
    network = GatedConvolution2DNetwork(input_channels=c_in, hidden_channels=c_hidden)
    output_tensor = network(input_tensor)
    assert output_tensor.shape == (1, c_out, hight, width), "Output shape mismatch!"
