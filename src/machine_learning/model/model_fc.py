"""Class for fully-connected network."""
import torch
from torch import Tensor, nn
from torch.nn import Linear


class LinearFCNetwork(nn.Module):
    """Linear FC architecture."""

    def __init__(
        self,
        num_spokes: int,
        num_readouts: int,
        im_w: int,
        calculate_magnitude: bool = True,
    ) -> None:
        """Initialize fully-connected block."""
        super().__init__()

        self.in_features = num_spokes * num_readouts
        self.out_features = im_w * im_w
        self.out_shape = (im_w, im_w)

        self.fc_block = Linear(
            in_features=self.in_features, out_features=self.out_features, bias=False
        )
        self.calculate_magnitude = calculate_magnitude

    def forward(self, x: Tensor) -> Tensor:
        """Pass a tensor through the module."""
        batch_size, k1, k2 = x.shape
        # reshape input so that the real and imaginary part of the k-space are stacked in the batch dimension
        x = x.view(batch_size * 2, k2)
        # pass input to machine learning model
        x = self.fc_block(x)
        # reshape output to a complex-valued quadratic matrix
        x = x.view(batch_size, 2, *self.out_shape)
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.view_as_complex(x)
        # calculate 2D inverse FFT
        x = torch.fft.ifft2(x, norm='forward')
        # calculate magnitude image
        if self.calculate_magnitude:
            x = torch.abs(x)
        # stack real and imaginay part in the channel dimension
        else:
            x = torch.stack((torch.real(x), torch.imag(x)), dim=1)
        return x
