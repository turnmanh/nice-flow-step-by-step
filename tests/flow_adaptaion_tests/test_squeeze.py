import pytest
import torch

from pytest_lazy_fixtures import lf

from src.flow.flow_adaptations import SqueezeFlow


@pytest.fixture()
def sample_input_tensor() -> torch.Tensor:
    """Fixture to create a sample tensor for testing."""
    batch_shape, channel_shape, height, width = 1, 1, 28, 28
    return (
        torch.arange(batch_shape * channel_shape * height * width)
        .reshape(batch_shape, channel_shape, height, width)
        .float()
    )


@pytest.mark.parametrize(
    "sample_tensor, squeezing_factor",
    [
        (lf("sample_input_tensor"), 2),
        (lf("sample_input_tensor"), 4),
    ],
)
def test_squeeze_flow(sample_tensor: torch.Tensor, squeezing_factor: int) -> None:
    batch_shape, channel_shape, height, width = sample_tensor.shape
    log_det_jacobian = torch.zeros(batch_shape)

    squeeze_flow = SqueezeFlow(squeeze_factor=squeezing_factor)
    squeezed, _ = squeeze_flow(sample_tensor, log_det_jacobian, reverse=False)    
    un_squeezed, _ = squeeze_flow(squeezed, log_det_jacobian, reverse=True)

    assert squeezed.shape == (
        batch_shape,
        channel_shape * (squeezing_factor**2),
        height // squeezing_factor,
        width // squeezing_factor,
    ), (
        f"Expected shape {(batch_shape, channel_shape * (squeezing_factor ** 2), height // squeezing_factor, width // squeezing_factor)}, "
        f"but got {squeezed.shape}"
    )
    assert (
        un_squeezed.shape == sample_tensor.shape
    ), f"Expected shape of original tensor {sample_tensor.shape}, but got {un_squeezed.shape}"
