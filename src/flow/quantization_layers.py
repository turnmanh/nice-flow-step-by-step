import torch
import torch.nn as nn


class Dequantization(nn.Module):

    def __init__(self, epsilon: float = 1e-5, strides: int = 256):
        """Implements a dequantization layer.

        The forward pass goes from the data space to the latent space by
        applying the dequantization transformation, followed by an
        inverse sigmoid transformation. The backward pass (reverse=True) goes
        from the latent space back to the data space by applying the sigmoid
        and inverse of quantization.

        Args:
            epsilon: Small value to prevent numerical instability.
            strides: Number of quantization levels.
        """
        super().__init__()
        self.epsilon = torch.tensor(epsilon)
        self.strides = torch.tensor(strides)

    def forward(
        self, x: torch.Tensor, log_det_jacobian: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if reverse:  # Data to latent space.
            z, log_det_jacobian = self.sigmoid(
                x=x, log_det_jacobian=log_det_jacobian, reverse=False
            )
            assert not torch.isnan(z).any(), (
                f"Data space tensor 'z' contains NaN values after sigmoid."
                f"Input was {z}."
            )
            z *= self.strides
            z = (
                torch.floor(z)
                .clamp(min=torch.tensor(0), max=self.strides - 1)
                .to(torch.int32)
            )
            log_det_jacobian += torch.log(self.strides) * z.shape[1:].numel()
        else:  # Latent space to data space.
            z, log_det_jacobian = self.dequantize(
                x=x, log_det_jacobian=log_det_jacobian
            )
            assert not torch.isnan(z).any(), (
                f"Latent space tensor 'z' contains NaN values after dequantization."
                f"Input was {z}."
            )
            z, log_det_jacobian = self.sigmoid(
                x=z, log_det_jacobian=log_det_jacobian, reverse=True
            )
            assert not torch.isnan(z).any(), (
                f"Latent space tensor 'z' contains NaN values after sigmoid."
                f"Input was {z}."
            )
        return z, log_det_jacobian

    def dequantize(
        self, x: torch.Tensor, log_det_jacobian: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dequantizes the input tensor using uniform noise and strides.

        Args:
            log_det_jacobian: Current log determinant of the Jacobian.
            x: Input tensor.

        Returns:
            Dequantized tensor and log determinant.
        """
        x = x.to(torch.float32)  # Convert to float32 for dequantization. Otherwise, the next line will cause an error.
        x += torch.rand_like(x).detach()
        x /= self.strides

        # Explanation: we dequantize the image by adding uniform noise to every
        # pixel value. Therefore, the change in volume is applied to each pixel.
        # In essence, each pixel is expanded to a small hypercube in the
        # continuous space. Therefore, the change on volume per pixel has to be
        # scaled to the data dimensions, i.e. all but the batch dimension. As
        # the dequantization is an expansion of the volume, the log determinant
        # has to be reduced by the volume change.
        log_det_jacobian -= torch.log(self.strides) * x.shape[1:].numel()
        return x, log_det_jacobian

    def sigmoid(
        self, x: torch.Tensor, log_det_jacobian: torch.Tensor, reverse: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply a sigmoid function to the input tensor.

        In order to ease the transformation between input values and targeted
        Gaussian, the sigmoid function is applied to compress the input values
        to the right range. Furthermore, the intermediate distribution is closer
        to a Gaussian.

        Returns:
            Sigmoid tensor and log determinant.
        """
        if reverse:
            # The input x is transformed (quantized) to be in range [0, 1).
            # To prevent issues in the logit, we scale the input to be in
            # range [0.5 * eps, 1- (0.5 * eps)).
            z = x * (1 - self.epsilon) + 0.5 * self.epsilon
            # The log determinant of the Jacobian is first updated by the
            # scaling of the input values.
            log_det_jacobian += torch.log(1 - self.epsilon) * z.shape[1:].numel()
            # The log determinant of the Jacobian is then updated by the
            # logit transformation.
            log_det_jacobian += (-1 * (torch.log(z) + torch.log(1 - z))).sum(
                dim=[1, 2, 3]
            )
            # We want the range of z to be the full reals; Therefore, the logit
            # is applied to transform the range to (-inf, inf).
            z = torch.log(z) - torch.log(1 - z)
        else:
            # The inverse of the logit transform is the sigmoid function.
            z = torch.sigmoid(x)
            # The inverse of the above scaling function.
            z = (z - 0.5 * self.epsilon) / (1 - self.epsilon)
            # The log derivative of the inverse of the logit transform,
            # the sigmoid, would be log(sig(x)(1 - sig(x))). However, a more
            # stable variant is using the Softplus function.
            log_det_jacobian += (-1 * x - 2 * torch.nn.functional.softplus(-x)).sum(
                dim=[1, 2, 3]
            )
            log_det_jacobian += -1 * torch.log(1 - self.epsilon) * z.shape[1:].numel()
        return z, log_det_jacobian


class VariationalDequantization(Dequantization):

    def __init__(self, variational_flows: list[nn.Module], epsilon: float = 1e-5):
        """Implements a variational dequantization layer.

        Instead of using a fixed uniform distribution, the dequantization layer
        learns a normalizing flow to model the dequantization distribution.

        Args:
            variational_flows: List of variational flows.
            epsilon: Small value to prevent numerical instability.
        """
        super().__init__(epsilon=epsilon)
        self.variational_flows = nn.ModuleList(variational_flows)

    def dequantize(
        self, x: torch.Tensor, log_det_jacobian: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = x.to(torch.float32)
        image = (z / self.strides) * 2 - 1  # Normalize to [-1, 1].

        # Compute the noise to add.
        noise = torch.randn_like(z)
        # Apply the inv. sigmoid, the logit transform, first to increase the
        # latent space to (-inf, inf).
        noise, log_det_jacobian = self.sigmoid(
            x=noise, log_det_jacobian=log_det_jacobian, reverse=True
        )

        for flow in self.variational_flows:
            # Learn the dequantization distribution using normalizing flows,
            # conditioned on the input image.
            noise, log_det_jacobian = flow(
                noise, log_det_jacobian, reverse=False, original_image=image
            )
        noise, log_det_jacobian = self.sigmoid(
            x=noise, log_det_jacobian=log_det_jacobian, reverse=False
        )

        # Apply dequantization with computed noise.
        # We have to add to the strides as we've added noise to the image.
        z = (z + noise) / (self.strides + 1)
        log_det_jacobian -= torch.log(self.strides + 1) * z.shape[1:].numel()
        return z, log_det_jacobian
