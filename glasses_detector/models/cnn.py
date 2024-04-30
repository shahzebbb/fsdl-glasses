"""Basic convolutional model"""

import argparse
import math 
from typing import Any, Dict

import torch
from torch import nn
import torch.nn.functional as F

CONV_LAYERS = 1
CONV_DIM = 64
CONV_KERNEL_SIZE = 3
CONV_STRIDE = 1
FC_DIM = 128
FC_DROPOUT = 0.25



class ConvBlock(nn.Module):
    """
    Simple 3x3 conv with padding size 1 (to leave the input size unchanged, followed by a ReLU.)
    """

    def __init__(self, input_size: int, input_channels: int, output_channels: int, kernel_size: int, stride: int) -> None:
        super().__init__()
        padding = math.ceil(((stride-1)*input_size - stride + kernel_size) / 2)
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the ConvBlock to x.
        
        Parameters
        ----------
        
        - x : (B, C, H, W) tensor
        
        Returns
        -------
        
        - torch.Tensor (B, C, H, W)
        """

        x = self.conv(x)
        r = self.relu(x)
        return r

class CNN(nn.Module):
    """Simple CNN for image recognition in a square image."""

    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None):
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config

        input_channels, input_height, input_width = self.data_config["input_dims"]
        assert (
            input_height == input_width
        ), f"input height and width should be equal, but was {input_height}, {input_width}"
        self.input_height, self.input_width = input_height, input_width

        num_classes = len(self.data_config["mapping"])

        conv_layers = self.args.get("conv_layers", CONV_LAYERS)
        conv_dim = self.args.get("conv_dim", CONV_DIM)
        conv_kernel_size = self.args.get("conv_kernel_size", CONV_KERNEL_SIZE)
        conv_stride = self.args.get("conv_stride", CONV_STRIDE)
        fc_dim = self.args.get("fc_dim", FC_DIM)
        fc_dropout = self.args.get("fc_dropout", FC_DROPOUT)

        self.conv1 = ConvBlock(input_height, input_channels, conv_dim, conv_kernel_size, conv_stride)
        conv_layers = [ConvBlock(input_height, conv_dim, conv_dim, conv_kernel_size, conv_stride) 
               for _ in range(conv_layers)]
        self.conv_layers = nn.Sequential(*conv_layers)
        self.dropout = nn.Dropout(fc_dropout)
        self.max_pool = nn.MaxPool2d(2)

        # Because our 3x3 convs have padding size 1, they leave the input size unchanged
        # The 2x2 maxpool divides the input size by 2
        conv_output_height, conv_output_width = input_height // 2, input_width // 2
        self.fc_input_dim = int(conv_output_height * conv_output_width * conv_dim)
        self.fc1 = nn.Linear(self.fc_input_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the CNN to x.
        
        Parameters
        -----------
        x
            (B, Ch, H, W) tensor, where H and W must equal input height and width from data_config.
            
            
        Returns
        -------
        torch.Tensor
            (B, Cl) tensor
        """

        B, Ch, H, W = x.shape
        assert H == self.input_height and W == self.input_width, f"bad inputs to CNN with shape {x.shape}"
        x = self.conv1(x) # B, CONV_DIM, H, W
        x = self.conv_layers(x) # B, CONV_DIM, H, W
        x = self.max_pool(x) # B, CONV_DIM, H // 2, W // 2
        x = self.dropout(x)
        x = torch.flatten(x, 1) # B, CONV_DIM * H // 2 * W // 2
        x = self.fc1(x) # B, FC_DIM
        x = F.relu(x)
        x = self.fc2(x) # B, Cl
        return x
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--conv_layers", type=int, default=CONV_LAYERS)
        parser.add_argument("--conv_dim", type=int, default=CONV_DIM)
        parser.add_argument("--conv_kernel_size", type=int, default=CONV_KERNEL_SIZE)
        parser.add_argument("--conv_stride", type=int, default=CONV_STRIDE)
        parser.add_argument("--fc_dim", type=int, default=FC_DIM)
        parser.add_argument("--fc_dropout", type=float, default=FC_DROPOUT)
        return parser
