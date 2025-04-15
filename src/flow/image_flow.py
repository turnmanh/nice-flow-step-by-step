import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

from src.flow.utils import compute_bpd


class ImageFlow(pl.LightningModule):

    def __init__(
        self,
        flows: list[nn.Module],
        prior: distributions.Distribution = distributions.normal.Normal(
            loc=0.0, scale=1.0
        ),
        k_fold_test_samples: int = 16,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """Implements a ImageFlow.

        Args:
            flows: List of flow modules.
            prior: Base distribution for the flow.
            device: Device to run the flow on.
            k_fold_test_samples: Number of times to estimate the ll on the test data. Defaults to 16.
        """
        super().__init__()
        self.flows = nn.ModuleList(flows)
        self.prior = prior
        # self.device = device
        self.k_fold_test_samples = k_fold_test_samples

    def forward(self, x: torch.Tensor, bpd: bool = True) -> torch.Tensor:
        """Performs forward pass through the flow.

        Args:
            x: Input data tensor.
            bpd: Whether to return bits per dimension.

        Returns:
            Loss tensor in the latent space; either NLL or BPD.
        """
        z, log_det_jacobian = self.to_latent_space(x)

        log_pz = self.prior.log_prob(z).sum(dim=[1, 2, 3])
        log_px = log_pz + log_det_jacobian

        # Obtain the loss.
        loss = -log_px

        # Return bits per dimension as loss, if required.
        if bpd:
            # Compute the bits per dimension and normalize by the number of non-batch dimensions.
            loss = compute_bpd(loss, torch.tensor(x.shape[1:]))
            loss = loss.mean()

        return loss

    def to_latent_space(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Transforms the input data to the latent space.

        Args:
            x: Input data tensor.

        Returns:
            Tuple of latent space tensor and log determinant.
        """
        # Initialize latent space and log determinant.
        z, log_det_jacobian = x, torch.zeros(x.size(0), device=x.device)

        # Iterate over transformations.
        for flow in self.flows:
            # Transform the prior sample z and the according log determinant.
            # Each flow adapts both values during transformation.
            z, log_det_jacobian = flow(z, log_det_jacobian, reverse=False)
            # Assert that z does not contain NaN values.
            assert not torch.isnan(z).any(), (
                f"Latent space tensor 'z' contains NaN values after {flow}."
                f"Input was {z}."
                f"Initial input was {x}."
            ) 


        return z, log_det_jacobian

    @torch.no_grad()
    def sample(self, num_samples: int) -> torch.Tensor:
        """Samples from the flow.

        Args:
            num_samples: Number of samples to generate.

        Returns:
            Sampled data tensor.
        """
        z = self.prior.sample((num_samples,)).to(self.device)

        log_det_jacobian = torch.zeros(z.size(0), device=z.device)
        for flow in reversed(self.flows):
            # Transform the prior sample z and the according log determinant.
            # Each flow adapts both values during transformation.
            z, log_det_jacobian = flow(z, log_det_jacobian, reverse=True)

        return z

    def configure_optimizers(self):
        """Defines the optimizer and learning rate scheduler."""
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def training_step(self, batch, batch_idx):
        """Performs a training step."""
        loss = self.forward(batch[0])
        self.log("train_loss (bpd)", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a validation step."""
        loss = self.forward(batch[0])
        self.log("val_loss (bpd)", loss)
        return loss

    def test_step(self, batch, batch_idx) -> None:
        """Perform a testing step after training.

        The likelihood is estimated on the full batch using the importance
        sampling method with a total of self.k_fold_test_samples samples for
        each item in the batch.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
        """
        samples = []
        for _ in range(self.k_fold_test_samples):
            # Obtain the NLL for each sample.
            samples.append(self.forward(batch[0], bpd=False))

        samples = torch.stack(samples, dim=-1)

        # Compute the average of log likelihood. To prevent numerical instability,
        # we use the logsumexp trick to sum the log likelihoods.
        ll = torch.logsumexp(samples, dim=-1) - torch.log(
            torch.tensor(self.k_fold_test_samples)
        )

        # Obtain the bits per dimension from NLL.
        ll_bpd = compute_bpd(ll, torch.tensor(batch[0].shape[1:]))
        ll_bpd = ll_bpd.mean()

        self.log("test_loss (bpd)", ll_bpd)
