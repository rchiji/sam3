# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Necks are the interface between a vision backbone and the rest of the detection model"""

from copy import deepcopy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from sam3.model.vitdet import ViT
from sam3.model.position_encoding import PositionEmbeddingSine


class Sam3DualViTDetNeck(nn.Module):
    def __init__(
        self,
        trunk: ViT,
        position_encoding: PositionEmbeddingSine,
        d_model: int,  # 256
        scale_factors: tuple[float] = (4.0, 2.0, 1.0, 0.5),
        add_sam2_neck: bool = False,
    ):
        """
        SimpleFPN neck a la ViTDet
        (From detectron2, very lightly adapted)
        It supports a "dual neck" setting, where we have two identical necks (for SAM3 and SAM2), with different weights

        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        """
        super().__init__()
        self.trunk: ViT = trunk
        self.position_encoding: PositionEmbeddingSine = position_encoding
        self.convs: nn.ModuleList = nn.ModuleList()

        self.scale_factors: tuple[float] = scale_factors
        use_bias: bool = True
        dim: int = self.trunk.channel_list[-1]

        for _, scale in enumerate(scale_factors):
            current: nn.Sequential = nn.Sequential()

            if scale == 4.0:
                current.add_module(
                    "dconv_2x2_0",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                current.add_module(
                    "gelu",
                    nn.GELU(),
                )
                current.add_module(
                    "dconv_2x2_1",
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                )
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module(
                    "dconv_2x2",
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                )
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module(
                    "maxpool_2x2",
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            current.add_module(
                "conv_1x1",
                nn.Conv2d(
                    in_channels=out_dim,
                    out_channels=d_model,
                    kernel_size=1,
                    bias=use_bias,
                ),
            )
            current.add_module(
                "conv_3x3",
                nn.Conv2d(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                ),
            )
            self.convs.append(current)

        self.sam2_convs = None
        if add_sam2_neck:
            # Assumes sam2 neck is just a clone of the original neck
            self.sam2_convs = deepcopy(self.convs)

    def forward(self, tensor_list: torch.Tensor) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        list[torch.Tensor] | None,
        list[torch.Tensor] | None,
    ]:
        # ViTを通す
        xs: list[torch.Tensor] = self.trunk(tensor_list)

        sam3_out, sam3_pos = [], []
        sam2_out, sam2_pos = None, None
        if self.sam2_convs is not None:
            sam2_out, sam2_pos = [], []

        # 最後のblockの出力を取得
        x: torch.Tensor = xs[-1]  # simpleFPN
        for i in range(len(self.convs)):
            sam3_x_out: torch.Tensor = self.convs[i](x)
            sam3_pos_out: torch.Tensor = self.position_encoding(sam3_x_out).to(sam3_x_out.dtype)
            sam3_out.append(sam3_x_out)
            sam3_pos.append(sam3_pos_out)

            if self.sam2_convs is not None:
                sam2_x_out = self.sam2_convs[i](x)
                sam2_pos_out = self.position_encoding(sam2_x_out).to(sam2_x_out.dtype)
                sam2_out.append(sam2_x_out)
                sam2_pos.append(sam2_pos_out)

        return sam3_out, sam3_pos, sam2_out, sam2_pos
