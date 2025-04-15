import torch
import torch.nn as nn


class ConcatenatedELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the Concatenated ELU activation function.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the activation function.
        """
        # Concatenate the ELU and its negative on the channel dimension.
        return torch.cat([nn.functional.elu(x), nn.functional.elu(-x)], dim=1)


class LayerNormChannels(nn.Module):

    def __init__(self, input_channels: int, eps: float = 1e-5):
        """Implements a Layer Normalization layer.

        Args:
            input_channels: Number of input channels.
            eps: Small value to avoid division by zero.
        """
        super().__init__()
        self.input_channels = input_channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, self.input_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, self.input_channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the Layer Normalization layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying Layer Normalization.
        """
        # Compute the mean and standard deviation across the channel dimension.
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)

        # Normalize the input tensor.
        x = (x - mean) / (std + self.eps)

        # Scale and shift the normalized tensor.
        x = self.gamma * x + self.beta

        return x


class GatedConvolution(nn.Module):

    def __init__(self, input_channels: int, hidden_channels: int):
        """Implements a Gated Convolution layer.

        Args:
            input_channels: Number of input channels.
            hidden_channels: Number of hidden channels.
        """
        super().__init__()
        self.net = nn.Sequential(
            ConcatenatedELU(),
            nn.Conv2d(2 * input_channels, hidden_channels, kernel_size=3, padding=1),
            ConcatenatedELU(),
            nn.Conv2d(
                2 * hidden_channels, 2 * input_channels, kernel_size=3, padding=1
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the Gated Convolution layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the Gated Convolution.
        """
        # Apply the gated convolution.
        z = self.net(x)
        value, gate = z.chunk(2, dim=1)
        # Apply the sigmoid activation function to the output.
        x = x + value * torch.sigmoid(gate)

        return x


class GatedConvolution2DNetwork(nn.Module):

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        output_channels: int = -1,
        num_layers: int = 3,
    ):
        """Implements a Gated Convolution layer.

        Args:
            input_channels: Number of input channels.
            hidden_channels: Number of hidden channels.
            output_channels: Number of output channels. If -1, it is set to
                2 * input_channels due to AFFINE transformation.
            num_layers: Number of layers in the network.
        """
        super().__init__()
        output_channels = self.set_output_channels(
            output_channels=output_channels, input_channels=input_channels
        )
        self.net = self.create_network(
            input_channels, hidden_channels, output_channels, num_layers
        )

    @staticmethod
    def set_output_channels(output_channels: int, input_channels: int) -> int:
        """Sets the number of output channels.

        Note, we're using AFFINE coupling. Therefore, we're estimating a scale
        and translation on a the dimension of the input. Therefore - at least in
        the default case - the output has to be twice the size / channels of the
        input.
        """
        if output_channels == -1:
            ret = 2 * input_channels
        else:
            ret = output_channels
        return ret

    @staticmethod
    def create_network(
        input_channels: int,
        hidden_channels: int,
        output_channels: int,
        num_layers: int,
    ) -> nn.Module:
        """Creates the Gated Convolution network."""
        layers = []

        # Create the first layer.
        layers.append(
            nn.Conv2d(input_channels, hidden_channels, kernel_size=3, padding=1)
        )

        for _ in range(num_layers):
            layers.append(GatedConvolution(hidden_channels, hidden_channels))
            layers.append(LayerNormChannels(hidden_channels))

        # Create the last layer.
        layers.append(ConcatenatedELU())
        layers.append(
            nn.Conv2d(2 * hidden_channels, output_channels, kernel_size=3, padding=1)
        )

        # Initialize the weights of the last stage in the network to 0 and 1.
        # This is done to ensure that the last stage produces values similar to
        # the identity map, in the beginning of the training.
        layers[-1].weight.data.zero_()
        layers[-1].bias.data.zero_()

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Computes the forward pass of the Gated Convolution layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor after applying the Gated Convolution.
        """
        # Apply the gated convolution.
        return self.net(x)
