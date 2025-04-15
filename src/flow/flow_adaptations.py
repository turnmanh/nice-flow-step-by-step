import torch
import torch.nn as nn


class SqueezeFlow(nn.Module):

    def __init__(self, squeeze_factor: int = 2):
        super().__init__()
        self.squeeze_factor = squeeze_factor

    def forward(
        self, x: torch.Tensor, log_det_jacobian: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies the squeeze operation to the input tensor.

        In order to reduce the size of the latent space with images, the image's
        spatial dimensions are reduced by shifting some of it to the channels.
        """
        batch_size, channels, height, width = x.size()
        
        if reverse:
            # Reverse the squeeze operation.
            z = x.reshape(
                batch_size,
                channels // (self.squeeze_factor**2),
                self.squeeze_factor,
                self.squeeze_factor,
                height,
                width,
            )
            z = z.permute(0, 1, 4, 2, 5, 3)
            z = z.reshape(
                batch_size,
                channels // (self.squeeze_factor**2),
                height * self.squeeze_factor,
                width * self.squeeze_factor,
            )
        else:
            # Make sure the height and width are divisible by the squeeze factor.
            assert (
                height % self.squeeze_factor == 0
            ), f"Height {height} must be divisible by squeeze factor {self.squeeze_factor}."

            assert (
                width % self.squeeze_factor == 0
            ), f"Width {width} must be divisible by squeeze factor {self.squeeze_factor}."

            # Apply the squeeze operation
            z = x.reshape(
                batch_size,
                channels,
                height // self.squeeze_factor,
                self.squeeze_factor,
                width // self.squeeze_factor,
                self.squeeze_factor,
            )
            z = z.permute(0, 1, 3, 5, 2, 4)
            z = z.reshape(
                batch_size,
                (self.squeeze_factor**2) * channels,
                height // self.squeeze_factor,
                width // self.squeeze_factor,
            )

        return z, log_det_jacobian
