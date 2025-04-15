import torch
import torchvision

from torchvision import transforms


def discretize_image(
    image: torch.Tensor, type: torch.dtype = torch.int32
) -> torch.Tensor:
    """Scales [0,1] image to [0,255] and converts to integer.

    Args:
        image: Input tensor to be discretized.
        type: Integer type for discretization.

    Returns:
        Discretized image.
    """
    assert (
        image.min() >= 0 and image.max() <= 1
    ), "Invalid image range. Expecting [0,1]."

    assert type in [
        torch.int16,
        torch.int32,
        torch.int64,
    ], "Invalid type for discretization. Expecting int."

    return (image * 255).type(type)


def split_data(
    data: torch.Tensor, split_ratio: float = 0.8
) -> tuple[torch.Tensor, torch.Tensor]:
    """Splits dataset into train and validation data.

    Args:
        data: Full dataset to be split of shape (N,*).
        split_ratio: Ratio of training data w.r.t. N.

    Returns:
        Tuple of training and validation data.
    """
    assert split_ratio > 0. and split_ratio < 1., "Invalid split ratio."

    data_size = len(data)
    split_size = data_size * split_ratio
    data_train, data_val = torch.utils.data.random_split(
        data, [int(split_size), int(data_size - split_size)]
    )

    return data_train, data_val


def get_data_loader(
    data: torch.Tensor,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    **kwargs,
) -> torch.utils.data.DataLoader:
    """Creates a data loader for the given dataset.

    Args:
        data: Dataset to be loaded.
        batch_size: Number of samples per batch.
        shuffle: If True, shuffles the data.
        num_workers: Number of subprocesses to use for data loading.

    Returns:
        DataLoader object for the given dataset.
    """
    return torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, **kwargs
    )


def load_mnist_data(
    root_data_path: str = "./datasets",
    transformations: transforms = transforms.Compose(
        [transforms.ToTensor(), discretize_image]
    ),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads MNIST dataset.

    Returns:
        MNIST datasets for training and testing.
    """
    mnist_data_train = torchvision.datasets.MNIST(
        root=root_data_path, train=True, download=True, transform=transformations
    )
    mnist_data_test = torchvision.datasets.MNIST(
        root=root_data_path, train=False, download=True, transform=transformations
    )
    return mnist_data_train, mnist_data_test

