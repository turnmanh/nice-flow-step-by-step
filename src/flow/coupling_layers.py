import torch
import torch.nn as nn


class CouplingLayer(nn.Module):

    def __init__(self, network: nn.Module, mask: torch.Tensor, input_channels: int = 3):
        """Implements a coupling layer.

        In the following logic, we apply a mask to the full input tensor. The
        mask itself dictates which parts are altered and which remain the same.

        Args:
            network: Neural network that parameterizes the coupling layer. mask:
            Binary mask that determines which features are transformed.
            input_channels: Number of input channels.
        """
        super().__init__()
        self.network = network
        self.scaling_factor = nn.Parameter(torch.zeros(input_channels))
        # The mask belongs to the state of the module, while being persistent.
        # As we want to serialize the buffer, e.g. when the model is saved, the
        # mask has to be added to the module's state.
        self.register_buffer("mask", mask)

    def forward(
        self,
        z: torch.Tensor,
        log_det_jacobian: torch.Tensor,
        reverse: bool = False,
        original_image: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the forward pass of the coupling layer.

        Args:
            z: Intermediate latent variable.
            log_det_jacobian: Current log determinant of the Jacobian.
            reverse: Direction of flow.
            original_image: Original image provided in case of Variational Dequantization.

        Returns:
            Latent variable after transformation and log determinant.
        """
        # Mask the input tensor. Note, 0 indicates the part to be transformed.
        # Therefore, the parts defining the transformation are contained using
        # 1 while the rest is zeroed out.
        z_input = z * self.mask

        # Decide whether to Variational Dequantization was used.
        if original_image is None:
            # Compute the AFFINE transformation.
            z_output = self.network(z_input)
        else:
            # Catenate the original image with the masked input along the channels.
            # Then, compute the AFFINE transformation.
            z_output = self.network(torch.cat([z_input, original_image], dim=1))

        # Split the transformation. Note, the transformation is AFFINE.
        # Therefore, the network yields a scale and a translation. 
        scale, translation = z_output.chunk(2, dim=1)

        # Enlarge the scaling factor to the full input tensor.
        scaling_factor = self.scaling_factor.exp().view(1, -1, 1, 1)
        # Apply the scaling factor to the signal. 
        scale = torch.tanh(scale / scaling_factor) * scaling_factor

        # Mask outputs to only transform the second part input.
        # Note, in our case, the mask is binary where 0 indicates the part to be transformed.
        scale = scale * (1 - self.mask)
        translation = translation * (1 - self.mask)

        # Apply the AFFINE transformation.
        if not reverse:
            z = z * torch.exp(scale) + translation
            # Update the log determinant of the Jacobian. Sum over all
            # dimensions except the batch dimension. The log determinant of the
            # Jacobian is the sum of the scaling factors. As the operations are
            # independent, the Jacobian is diagonal. Furthermore, the only
            # elements of the log Jacobian are the scaling factors in scale.
            # Therefore, a summation of all such values suffices.
            log_det_jacobian += scale.sum(dim=[1, 2, 3])
        else:
            # Apply the inverse transformation.
            z = (z - translation) * torch.exp(-scale)
            # Update the log determinant of the Jacobian. The log determinant of
            # the Jacobian is the sum of the scaling factors. As we're inverting
            # the transformation, the sign has to be negative.
            log_det_jacobian -= scale.sum(dim=[1, 2, 3])

        return z, log_det_jacobian


