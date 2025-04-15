import os
import time
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.data.data import plot_image_samples, get_samples_from_data_loader
from src.data.utils import load_mnist_data, split_data, get_data_loader
from src.flow.coupling_layers import CouplingLayer
from src.flow.image_flow import ImageFlow
from src.flow.networks import GatedConvolution2DNetwork
from src.flow.quantization_layers import Dequantization, VariationalDequantization
from src.flow.utils import create_checkerboard_mask, create_channel_mask


# Define constants.
CHECKPOINT_PATH: str = os.path.join(os.path.dirname(__file__), "checkpoints")
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS_DATA: int = 4


def create_simple_flow(use_variational_dequantization=True):
    flow_layers = []
    if use_variational_dequantization:
        variational_dequantization_layers = [
            CouplingLayer(
                network=GatedConvolution2DNetwork(
                    input_channels=2, hidden_channels=16, output_channels=2
                ),
                mask=create_checkerboard_mask(shape=(28, 28), invert=(i % 2 == 1)),
                input_channels=1,
            )
            for i in range(4)
        ]
        flow_layers += [
            VariationalDequantization(
                variational_flows=variational_dequantization_layers
            )
        ]
    else:
        flow_layers += [Dequantization()]

    for i in range(8):
        flow_layers += [
            CouplingLayer(
                network=GatedConvolution2DNetwork(input_channels=1, hidden_channels=32),
                mask=create_checkerboard_mask(shape=(28, 28), invert=(i % 2 == 1)),
                input_channels=1,
            )
        ]

    flow_model = ImageFlow(flow_layers)
    return flow_model


def train_flow(flow, model_name="MNISTFlow"):
    # Create a PyTorch Lightning trainer
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
        accelerator="gpu", # if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=200,
        gradient_clip_val=1.0,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="min", monitor="val_bpd"),
            LearningRateMonitor("epoch"),
        ],
        check_val_every_n_epoch=5,
    )

    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    # Load MNIST data.
    train_data, test_data = load_mnist_data()
    train_data, val_data = split_data(train_data)

    # Print data shapes.
    print(f"Train data shape: {len(train_data)}")

    # Create data loaders.
    train_loader = get_data_loader(
        train_data, shuffle=True, num_workers=NUM_WORKERS_DATA, persistent_workers=True
    )
    val_loader = get_data_loader(
        val_data, shuffle=False, num_workers=NUM_WORKERS_DATA, persistent_workers=True
    )
    test_loader = get_data_loader(
        test_data, shuffle=False, num_workers=NUM_WORKERS_DATA, persistent_workers=True
    )

    result = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, model_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model, loading...")
        intermediate_checkpoint = torch.load(pretrained_filename, map_location=device)
        flow.load_state_dict(intermediate_checkpoint["state_dict"])
        result = intermediate_checkpoint.get("result", None)
    else:
        print("Start training", model_name)
        trainer.fit(flow, train_loader, val_loader)

    # Test best model on validation and test set if no result has been found
    # Testing can be expensive due to the importance sampling.
    if result is None:
        val_result = trainer.test(flow, val_loader, verbose=False)
        start_time = time.time()
        test_result = trainer.test(flow, test_loader, verbose=False)
        duration = time.time() - start_time
        result = {
            "test": test_result,
            "val": val_result,
            "time": duration / len(test_loader) / flow.import_samples,
        }

    return flow, result


def main():
    print(f"Using device: {device}")
    flow_model = create_simple_flow(use_variational_dequantization=False)
    flow_model, result = train_flow(flow_model, model_name="MNISTFlow")
    print(f"Test result: {result['test']}")
    print(f"Validation result: {result['val']}")
    print(f"Time per sample: {result['time']}")


if __name__ == "__main__":
    # Create the checkpoint directory if it doesn't exist.
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    # Set the compute device.
    device = torch.device(DEVICE)

    main()
