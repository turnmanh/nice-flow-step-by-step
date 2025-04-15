import torch
import torch.nn as nn

from toch.distributions import Distribution


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


class SplitFlow(nn.Module):

    def __init__(
        self,
        split_factor: int = 2,
        prior_dist: Distribution = torch.distributions.Normal(0.0, 1.0),
    ):
        """Initialize the SplitFlow class.

        Args:
            split_factor: The factor by which to split the input tensor.
            prior_dist: The prior distribution to use for the split flow.
        """ 
        super().__init__()
        self.split_factor = split_factor
        self.prior_dist = prior_dist

    @split_factor.setter
    def split_factor(self, value: int) -> None:
        """Set the split factor."""
        if value <= 0:
            raise ValueError("Split factor must be a positive integer.")
        if value != 2:
            raise NotImplementedError("Only split factor of 2 is implemented.")
        self._split_factor = value

    @property
    def split_factor(self) -> int:
        """Get the split factor."""
        return self._split_factor

    def forward(
        self, x: torch.Tensor, log_det_jacobian: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies the split operation to the input tensor.

        This is used to split the inputs into parts where one part is directly
        evaluated against the prior and the other further transformed by
        coupling blocks.
        """
        if reverse:
            z_split = self.prior_dist.sample(sample_shape=x.shape)
            z = torch.cat((x, z_split), dim=1)
            log_det_jacobian -= self.prior_dist.log_prob(z_split).sum(dim=[1, 2, 3])
        else:
            # todo: Adapt for arbitrary split factor.
            z, z_split = x.chunk(self.split_factor, dim=1)
            log_det_jacobian += self.prior_dist.log_prob(z_split).sum(dim=[1, 2, 3])
        
        return z, log_det_jacobian

