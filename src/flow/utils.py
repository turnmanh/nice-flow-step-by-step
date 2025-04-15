import torch

from src.custom_tyes.masking_types import MaskingType


def compute_bpd(nll: torch.Tensor, dims: torch.Tensor) -> torch.Tensor:
    """Computes the bits per dimension (BPD) loss.

    Returns:
        BPD loss tensor.
    """
    loss = nll * torch.log2(torch.exp(torch.tensor(1.0))) / torch.prod(dims)

    return loss


def create_checkerboard_mask(
    shape: tuple[int, int], invert: bool = False, mask_type: MaskingType = MaskingType()
) -> torch.Tensor:
    """Creates a checkerboard mask for the given shape.

    Args:
        shape: Shape of the mask.
        invert: Whether to invert the mask. This is useful when stacking 
            the mask to alternate the pixels to which the mask is applied.
        mask_type: Type of the mask. Can be "checkerboard" or "stripes".

    Returns:
        Mask tensor.
    """
    if mask_type.mask == "checkerboard":
        mask = torch.zeros(shape)
        mask[0::2, 0::2] = 1
        mask[1::2, 1::2] = 1
    
    if mask_type.mask == "stripes":
        mask = torch.zeros(shape)
        mask[0::2, :] = 1

    if invert:
        mask = 1 - mask

    # Reshape the mask to match the network transformation shapes. 
    mask = mask.view(1, 1, *shape)

    return mask


def create_channel_mask(c_in: int, invert: bool = False) -> torch.Tensor:
    """Creates a channel mask for the given number of channels.

    Args:
        c_in: Number of input channels. 
        invert: Whether to invert the mask. This is useful when stacking 
            the mask to alternate the pixels to which the mask is applied.

    Returns:
        Channel mask tensor.
    """
    mask = torch.zeros(c_in)
    mask[0::2] = 1

    if invert:
        mask = 1 - mask

    # Reshape the mask to match the network transformation shapes. 
    mask = mask.view(1, c_in, 1, 1)

    return mask
