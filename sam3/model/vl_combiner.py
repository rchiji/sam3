# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Provides utility to combine a vision backbone with a language backbone."""

from copy import copy
from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend

from .act_ckpt_utils import activation_ckpt_wrapper
from .necks import Sam3DualViTDetNeck
from sam3.model.text_encoder_ve import VETextEncoder


class SAM3VLBackbone(nn.Module):
    """This backbone combines a vision backbone and a language backbone without fusion.
    As such it is more of a convenience wrapper to handle the two backbones together.

    It adds support for activation checkpointing and compilation.
    """

    def __init__(
        self,
        visual: Sam3DualViTDetNeck,
        text: VETextEncoder,
        compile_visual: bool = False,
        act_ckpt_whole_vision_backbone: bool = False,
        act_ckpt_whole_language_backbone: bool = False,
        scalp: int = 0,
    ):
        """Initialize the backbone combiner.

        :param visual: The vision backbone to use
        :param text: The text encoder to use

        scalp: Number of the lowest resolution features to discard from the vision backbone FPN output. 1だとscale=0.5が捨てられてscale=1以降が残る。
        """
        super().__init__()
        self.vision_backbone: Sam3DualViTDetNeck = torch.compile(visual) if compile_visual else visual
        self.language_backbone: VETextEncoder = text
        self.scalp: int = scalp
        # allow running activation checkpointing on the entire vision and language backbones
        self.act_ckpt_whole_vision_backbone = act_ckpt_whole_vision_backbone
        self.act_ckpt_whole_language_backbone = act_ckpt_whole_language_backbone

    def forward(
        self,
        samples: torch.Tensor,
        captions: list[str],
        input_boxes: torch.Tensor | None = None,
        additional_text: list[str] | None = None,
    ):
        """Forward pass of the backbone combiner.

        :param samples: The input images
        :param captions: The input captions
        :param input_boxes: If the text contains place-holders for boxes, this
            parameter contains the tensor containing their spatial features
        :param additional_text: This can be used to encode some additional text
            (different from the captions) in the same forward of the backbone
        :return: Output dictionary with the following keys:
            - vision_features: The output of the vision backbone
            - language_features: The output of the language backbone
            - language_mask: The attention mask of the language backbone
            - vision_pos_enc: The positional encoding of the vision backbone
            - (optional) additional_text_features: The output of the language
                backbone for the additional text
            - (optional) additional_text_mask: The attention mask of the
                language backbone for the additional text
        """
        # multiscale ViTを通す
        output: dict[str, torch.Tensor | None] = self.forward_image(
            samples,
        )
        device: torch.device = output["vision_features"].device
        output.update(
            # text encoder出力を追加
            self.forward_text(
                captions,
                input_boxes,
                additional_text,
                device,
            ),
        )
        return output

    def forward_image(self, samples: torch.Tensor):
        return activation_ckpt_wrapper(self._forward_image_no_act_ckpt)(
            samples=samples,
            act_ckpt_enable=self.act_ckpt_whole_vision_backbone and self.training,
        )

    def _forward_image_no_act_ckpt(
        self,
        samples: torch.Tensor,
    ):
        # Forward through backbone
        # multiscale ViTを通す
        sam3_features: list[torch.Tensor]  # [[1,256,288,288],[1,256,144,144],[1,256,72,72],[1,256,36,36]]
        sam3_pos: list[torch.Tensor]  # [[1,256,288,288],[1,256,144,144],[1,256,72,72],[1,256,36,36]]
        sam3_features, sam3_pos, sam2_features, sam2_pos = self.vision_backbone.forward(
            samples,
        )

        if self.scalp > 0:
            # Discard the lowest resolution features
            sam3_features, sam3_pos = (
                sam3_features[: -self.scalp],
                sam3_pos[: -self.scalp],
            )
            if sam2_features is not None and sam2_pos is not None:
                sam2_features, sam2_pos = (
                    sam2_features[: -self.scalp],
                    sam2_pos[: -self.scalp],
                )

        sam2_output = None

        if sam2_features is not None and sam2_pos is not None:
            sam2_src: torch.Tensor = sam2_features[-1]
            sam2_output = {
                "vision_features": sam2_src,
                "vision_pos_enc": sam2_pos,
                "backbone_fpn": sam2_features,
            }

        sam3_src = sam3_features[-1]
        output: dict[str, torch.Tensor | None] = {
            "vision_features": sam3_src,
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": sam2_output,
        }

        return output

    def forward_text(self, captions, input_boxes=None, additional_text=None, device="cuda"):
        return activation_ckpt_wrapper(self._forward_text_no_ack_ckpt)(
            captions=captions,
            input_boxes=input_boxes,
            additional_text=additional_text,
            device=device,
            act_ckpt_enable=self.act_ckpt_whole_language_backbone and self.training,
        )

    def _forward_text_no_ack_ckpt(
        self,
        captions,
        input_boxes=None,
        additional_text=None,
        device="cuda",
    ):
        output = {}

        # Forward through text_encoder
        text_to_encode = copy(captions)
        if additional_text is not None:
            # if there are additional_text, we piggy-back them into this forward.
            # They'll be used later for output alignment
            text_to_encode += additional_text

        sdpa_context = sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
            ]
        )

        with sdpa_context:
            text_attention_mask, text_memory, text_embeds = self.language_backbone(
                text_to_encode, input_boxes, device=device
            )

        if additional_text is not None:
            output["additional_text_features"] = text_memory[:, -len(additional_text) :]
            output["additional_text_mask"] = text_attention_mask[-len(additional_text) :]

        text_memory = text_memory[:, : len(captions)]
        text_attention_mask = text_attention_mask[: len(captions)]
        text_embeds = text_embeds[:, : len(captions)]
        output["language_features"] = text_memory
        output["language_mask"] = text_attention_mask
        output["language_embeds"] = text_embeds  # Text embeddings before forward to the encoder

        return output
