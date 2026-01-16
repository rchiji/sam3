# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
from typing import Dict, List

import numpy as np
import PIL
import torch

from sam3.model import box_ops
from sam3.model.sam3_image import Sam3Image
from sam3.model.data_misc import FindStage, interpolate
from torchvision.transforms import v2


class Sam3Processor:
    """ """

    def __init__(
        self,
        model: Sam3Image,
        resolution=1008,
        device="cuda",
        confidence_threshold=0.5,
    ):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.confidence_threshold = confidence_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(
        self,
        image: PIL.Image.Image | torch.Tensor | np.ndarray,
        state: dict | None = None,
    ) -> dict[str, any]:
        """
        Sets the image on which we want to do predictions.

        画像を登録し、バックボーン（SAM3VLBackbone）の特徴量を計算してstateに保存する。

        :param image: A PIL image or a tensor
        :param state: An optional state dictionary to store intermediate results
        :return: The updated state dictionary
        """
        if state is None:
            state: dict = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image: torch.Tensor = v2.functional.to_image(image).to(self.device)
        image: torch.Tensor = self.transform(image).unsqueeze(0)  # (1,3,1008,1008)

        state["original_height"] = height
        state["original_width"] = width

        # SAM3VLBackboneを通して、画像の特徴量リストと位置エンコードリスト、text特徴量などを取得しdictを得る。
        state["backbone_out"] = self.model.backbone.forward_image(image)

        # SAM1/SAM2実装モード
        inst_interactivity_en: bool = self.model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                sam2_backbone_out["backbone_fpn"][0]
            )
            sam2_backbone_out["backbone_fpn"][1] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                sam2_backbone_out["backbone_fpn"][1]
            )
        return state

    @torch.inference_mode()
    def set_image_batch(
        self,
        images: list[PIL.Image.Image],
        state: dict[str, any] | None = None,
    ):
        """Sets the image batch on which we want to do predictions."""
        if state is None:
            state: dict[str, any] = {}

        if not isinstance(images, list):
            raise ValueError("Images must be a list of PIL images or tensors")
        assert len(images) > 0, "Images list must not be empty"
        assert isinstance(images[0], PIL.Image.Image), "Images must be a list of PIL images"

        state["original_heights"] = [image.height for image in images]
        state["original_widths"] = [image.width for image in images]

        # resize and normalize
        images: list[torch.Tensor] = [
            self.transform(
                v2.functional.to_image(image).to(self.device),
            )
            for image in images
        ]
        images: torch.Tensor = torch.stack(images, dim=0)  # (B,3,1008,1008)

        # SAM3VLBackboneを通して、画像の特徴量リストと位置エンコードリスト、text特徴量などを取得しdictを得る。
        state["backbone_out"] = self.model.backbone.forward_image(images)
        inst_interactivity_en: bool = self.model.inst_interactive_predictor is not None

        # SAM1/SAM2実装モード
        if inst_interactivity_en and "sam2_backbone_out" in state["backbone_out"]:
            sam2_backbone_out = state["backbone_out"]["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                sam2_backbone_out["backbone_fpn"][0]
            )
            sam2_backbone_out["backbone_fpn"][1] = self.model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                sam2_backbone_out["backbone_fpn"][1]
            )
        return state

    @torch.inference_mode()
    def _add_text_prompt(self, prompt: str, state: Dict):
        """Adds a text prompt and encodes it without running inference."""
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before _add_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)

        # Create dummy geometric prompt and encode
        geometric_prompt = self.model._get_dummy_prompt()
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            prompt_encoded, prompt_mask, backbone_out = self.model._encode_prompt(
                backbone_out=state["backbone_out"], find_input=self.find_stage, geometric_prompt=geometric_prompt
            )

        state["prompt"] = prompt_encoded
        state["prompt_mask"] = prompt_mask
        state["backbone_out"] = backbone_out

        return state

    @torch.inference_mode()
    def _add_box_prompt(
        self,
        box: list[float],  # [x_center, y_center, width, height] normalized
        label: bool,  # True for foreground, False for background
        state: dict[str, any],
    ):
        """
        Adds a box prompt and encodes it without running inference.

        Note:
            self.find_stageがid:0しか持たないので、画像バッチ処理には対応していない。
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before _add_box_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(["visual"], device=self.device)
            state["backbone_out"].update(dummy_text_outputs)

        # Initialize geometric prompt if needed
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes: torch.Tensor = torch.tensor(box, device=self.device, dtype=torch.float32).view(
            1, 1, 4
        )  # (1,1,4) なので1 box, 1 batchでしか扱わない想定
        labels: torch.Tensor = torch.tensor([label], device=self.device, dtype=torch.bool).view(
            1, 1
        )  # (1,1) なので1 box, 1 batchでしか扱わない想定
        state["geometric_prompt"].append_boxes(boxes, labels)

        # forward_grounding内で行っていたprompt encodingをここで実行するように変更
        # Encode prompts immediately
        with torch.profiler.record_function("SAM3Image._encode_prompt"):
            prompt, prompt_mask, backbone_out = self.model._encode_prompt(
                backbone_out=state["backbone_out"],
                find_input=self.find_stage,
                geometric_prompt=state["geometric_prompt"],
            )

        # Store encoded prompts directly in state
        state["prompt"] = prompt
        state["prompt_mask"] = prompt_mask
        state["backbone_out"] = backbone_out

        # Remove geometric_prompt as we only rely on prompt and prompt_mask for inference

        # (260114) geometric_promptを削除しないようにコメントアウト
        # if "geometric_prompt" in state:
        #     del state["geometric_prompt"]

        return state

    def reset_all_prompts(self, state: Dict):
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "prompt", "prompt_mask", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            if key in state:
                del state[key]

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    # Public API for inference
    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: Dict):
        """Sets the text prompt and run the inference."""
        self._add_text_prompt(prompt, state)
        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(
        self,
        box: list,  # [x_center, y_center, width, height] normalized
        label: bool,  # True for foreground, False for background
        state: dict[str, any],
    ):
        """Adds a box prompt and run the inference."""

        # (260114) add_geometric_prompt内で行っていたprompt encodingを別関数化したので、ここでは1行の処理になっている。
        self._add_box_prompt(box, label, state)
        return self._forward_grounding(state)

    @torch.inference_mode()
    def _forward_grounding(
        self,
        state: dict[str, any],
    ) -> dict[str, any]:
        """Runs the grounding heads and updates the state with the results."""

        # Sam3Imageのforward_grounding
        # （transformer encoder -> transformer decoder -> segmentation headsの順にforward）
        outputs: dict[str, any] = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            prompt=state["prompt"],
            prompt_mask=state["prompt_mask"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state
