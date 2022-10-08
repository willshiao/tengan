from os import remove
import torch

def mask_torch_data(ten, frac=0.1, return_full_idxs=True, device='cpu', detach=True, existing_idxs=None):
    """Randomly masks out values from a N-dimensional Torch tensor.
    Creates a copy of the input tensor.

    Args:
        ten (torch.Tensor): The tensor to pull values from.
        frac (float, optional): The ratio of values to mask out. Defaults to 0.1.
        detach (bool, optional): Whether or not to allow autograd back to the input. Defaults to True.

    Returns:
        (torch.Tensor, torch.Tensor, torch.Tensor): Returns 3 values:
            1. The input array with values masked out.
            2. The values taken from the input array.
            3. The indices (when using np.flat indexing) where the values were taken.
    """
    n_el = torch.numel(ten)
    n_select = round(frac * n_el)
    if existing_idxs is None:
        remove_idxs = torch.randperm(n_el, device=device)[:n_select]
    else:
        remove_idxs = existing_idxs[:n_select]
    out = torch.clone(ten)

    if detach:
        out = out.detach()
    flat_out = out.view(-1)
    missing_vals = flat_out[remove_idxs]
    flat_out[remove_idxs] = 0
    return (out, missing_vals, remove_idxs)

def select_with_flat_idxs(ten, flat_idxs, with_batch=False):
    ten_flat = ten.view(-1)
    return ten_flat[flat_idxs]
