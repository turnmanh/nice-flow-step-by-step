import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot_image_samples(samples: torch.Tensor, labels: torch.Tensor) -> None:
    """Plots sample images from the dataset.

    Args:
        samples: Sample images to be plotted.
        labels: Corresponding labels for the images.
    """
    fig, axes = plt.subplots(3, 3, figsize=(6, 6))

    for i, ax in enumerate(axes.flat):
        ax.imshow(samples[i].permute(1, 2, 0))
        ax.set_title(f"Label: {labels[i].item()}")
        ax.axis("off")

    fig.tight_layout()
    plt.show()


def get_samples_from_data_loader(
    data_loader: torch.utils.data.DataLoader,
    num_samples: int = 9,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fetches sample images and labels from the data loader.

    Args:
        data_loader: DataLoader object containing the dataset.
        num_samples: Number of samples to fetch.

    Returns:
        Tuple of sample images and labels.
    """
    acc_samples, acc_labels = [], []
    for idx, (sample, label) in enumerate(data_loader):
        acc_samples.append(sample)
        acc_labels.append(label)

        if idx + 1 == num_samples:
            break

    samples = torch.cat(acc_samples)
    labels = torch.cat(acc_labels)

    return samples, labels