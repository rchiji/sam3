# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .model_misc import MLP


class LinearPresenceHead(nn.Sequential):
    def __init__(self, d_model):
        # a hack to make `LinearPresenceHead` compatible with old checkpoints
        super().__init__(nn.Identity(), nn.Identity(), nn.Linear(d_model, 1))

    def forward(self, hs, prompt, prompt_mask):
        return super().forward(hs)


class MaskPredictor(nn.Module):
    def __init__(self, hidden_dim, mask_dim):
        super().__init__()
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

    def forward(
        self,
        obj_queries: torch.Tensor,  # (1,100,256)
        pixel_embed: torch.Tensor,  # (1,256,288,288)
    ):
        if len(obj_queries.shape) == 3:
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = torch.einsum("bqc,chw->bqhw", self.mask_embed(obj_queries), pixel_embed)
            else:
                mask_preds = torch.einsum("bqc,bchw->bqhw", self.mask_embed(obj_queries), pixel_embed)
        else:
            # Assumed to have aux masks
            if pixel_embed.ndim == 3:
                # batch size was omitted
                mask_preds = torch.einsum("lbqc,chw->lbqhw", self.mask_embed(obj_queries), pixel_embed)
            else:
                mask_preds = torch.einsum("lbqc,bchw->lbqhw", self.mask_embed(obj_queries), pixel_embed)

        return mask_preds


class SegmentationHead(nn.Module):
    def __init__(
        self,
        hidden_dim,
        upsampling_stages,
        use_encoder_inputs=False,  # <-- encoder hidden statesを使用するかどうか
        aux_masks=False,
        no_dec=False,
        pixel_decoder=None,
        act_ckpt=False,
        shared_conv=False,
        compile_mode_pixel_decoder=None,
    ):
        super().__init__()
        self.use_encoder_inputs = use_encoder_inputs
        self.aux_masks = aux_masks
        if pixel_decoder is not None:
            self.pixel_decoder: nn.Module | PixelDecoder = pixel_decoder
        else:
            self.pixel_decoder: PixelDecoder = PixelDecoder(
                hidden_dim,
                upsampling_stages,
                shared_conv=shared_conv,
                compile_mode=compile_mode_pixel_decoder,
            )
        self.no_dec = no_dec
        if no_dec:
            self.mask_predictor: nn.Conv2d = nn.Conv2d(
                hidden_dim,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.mask_predictor: MaskPredictor = MaskPredictor(hidden_dim, mask_dim=hidden_dim)

        self.act_ckpt = act_ckpt

        # used to update the output dictionary
        self.instance_keys = ["pred_masks"]

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def to(self, *args, **kwargs):
        # clear cached _device in case the model is moved to a different device
        self._device = None
        return super().to(*args, **kwargs)

    def _embed_pixels(
        self,
        backbone_feats: list[torch.Tensor],
        image_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        (1,256,288,288)の特徴マップまでDecodeする。

        backbone_feats: SAM3VLBackbone.forward_image出力 [(1,3,288,288),(1,3,144,144),(1,256,72,72)]
        image_ids: (batch_size,) 各サンプルのbackbone_featsにおけるインデックス
        encoder_hidden_states: (spatial_dim*spatial_dim, batch_size, hidden_dim) Transformerエンコーダの出力 (5184,1,256)
        """

        feature_device: torch.device = backbone_feats[0].device  # features could be on CPU
        model_device = self.device
        image_ids_: torch.Tensor = image_ids.to(feature_device)

        # encoder_hidden_statesを使用する場合
        if self.use_encoder_inputs:
            if backbone_feats[0].shape[0] > 1:
                # For bs > 1, we construct the per query backbone features
                backbone_visual_feats: list[torch.Tensor] = []
                for feat in backbone_feats:
                    # Copy the img features per query (pixel decoder won't share img feats)
                    backbone_visual_feats.append(
                        feat[image_ids_, ...].to(model_device),
                    )
            else:
                # Bs=1, we rely on broadcasting for query-based processing
                backbone_visual_feats: list[torch.Tensor] = [bb_feat.clone() for bb_feat in backbone_feats]

            # Extract visual embeddings
            # encoder_hidden_states: (5184,1,256) を (1,256,72,72)に変換
            encoder_hidden_states = encoder_hidden_states.permute(1, 2, 0)  # (5184,1,256) -> (1,256,5184)
            spatial_dim: int = math.prod(backbone_feats[-1].shape[-2:])  # 72*72=5184
            encoder_visual_embed: torch.Tensor = encoder_hidden_states[..., :spatial_dim].reshape(
                -1,
                *backbone_feats[-1].shape[1:],  # (256,72,72)
            )  # (1,256,72,72)

            # SAM3VLBackbone.forward_imageの最後の要素をencoder_hidden_statesをreshapeした出力で置き換え
            backbone_visual_feats[-1] = encoder_visual_embed

            # この特徴マップをアップサンプリングしながら大きい特徴マップと加算 ➔ 畳み込み ➔ GroupNorm ➔ ReLU を行い[1,256,288,288]のpixel_embedを取得する。
            if self.act_ckpt:
                pixel_embed = checkpoint.checkpoint(self.pixel_decoder, backbone_visual_feats, use_reentrant=False)
            else:
                pixel_embed = self.pixel_decoder(backbone_visual_feats)
        else:
            backbone_feats = [x.to(model_device) for x in backbone_feats]
            pixel_embed = self.pixel_decoder(backbone_feats)
            if pixel_embed.shape[0] == 1:
                # For batch_size=1 training, we can avoid the indexing to save memory
                pixel_embed = pixel_embed.squeeze(0)
            else:
                pixel_embed = pixel_embed[image_ids, ...]

        return pixel_embed

    def forward(
        self,
        backbone_feats: list[torch.Tensor],
        obj_queries: torch.Tensor,
        image_ids,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        if self.use_encoder_inputs:
            assert encoder_hidden_states is not None

        pixel_embed: torch.Tensor = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )

        if self.no_dec:
            mask_pred = self.mask_predictor(pixel_embed)
        elif self.aux_masks:
            mask_pred = self.mask_predictor(obj_queries, pixel_embed)
        else:
            mask_pred = self.mask_predictor(obj_queries[-1], pixel_embed)

        return {"pred_masks": mask_pred}


class PixelDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_upsampling_stages: int,  # 3が指定されているが、実際はfeature mapの数-1回アップサンプリングされる
        interpolation_mode: str = "nearest",
        shared_conv: bool = False,
        compile_mode: str | None = None,
    ):
        super().__init__()
        self.hidden_dim: int = hidden_dim
        self.num_upsampling_stages: int = num_upsampling_stages
        self.interpolation_mode: str = interpolation_mode

        conv_layers: list[nn.Conv2d] = []
        norms: list[nn.GroupNorm] = []
        # 畳み込み回数。3が指定されているが、実際はfeature mapの数-1回アップサンプリングされる
        num_convs: int = 1 if shared_conv else num_upsampling_stages
        for _ in range(num_convs):
            conv_layers.append(
                nn.Conv2d(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                )
            )
            norms.append(nn.GroupNorm(8, self.hidden_dim))

        self.conv_layers: nn.ModuleList = nn.ModuleList(conv_layers)
        self.norms: nn.ModuleList = nn.ModuleList(norms)
        self.shared_conv: bool = shared_conv
        self.out_dim: int = self.conv_layers[-1].out_channels
        if compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=compile_mode, dynamic=True, fullgraph=True)
            # Needed to make checkpointing happy. But we don't know if the module is checkpointed, so we disable it by default.
            torch._dynamo.config.optimize_ddp = False

    def forward(
        self,
        backbone_feats: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            backbone_feats: List of feature maps from the backbone, ordered from
                highest to lowest resolution.

            SAM3VLBackbone.forward_image出力 [(1,3,288,288),(1,3,144,144),(1,256,72,72)]

        Returns:
            torch.Tensor: The output feature map after upsampling and fusion.

        """
        # Assumes backbone features are already projected (C == hidden dim)

        prev_fpn: torch.Tensor = backbone_feats[-1]  # (1,256,72,72)
        fpn_feats: list[torch.Tensor] = backbone_feats[:-1]  # [(1,3,288,288),(1,3,144,144)]

        ## アップサンプリングしながらSAM3VLBackbone出力を加算➔conv➔norm➔ReLU
        # [(1,3,288,288),(1,3,144,144)]の順
        for layer_idx, bb_feat in enumerate(fpn_feats[::-1]):
            curr_fpn: torch.Tensor = bb_feat
            # 小さい特徴マップを大きい特徴マップに合わせてアップサンプリングして加算
            prev_fpn: torch.Tensor = curr_fpn + F.interpolate(
                prev_fpn,
                size=curr_fpn.shape[-2:],
                mode=self.interpolation_mode,
            )
            # 畳み込み＋正規化＋ReLU
            if self.shared_conv:
                # only one conv layer
                layer_idx = 0
            prev_fpn: torch.Tensor = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn: torch.Tensor = F.relu(self.norms[layer_idx](prev_fpn))

        return prev_fpn


class UniversalSegmentationHead(SegmentationHead):
    """This module handles semantic+instance segmentation"""

    def __init__(
        self,
        hidden_dim: int,  # 256
        upsampling_stages: int,  # 3
        pixel_decoder: PixelDecoder,
        aux_masks: bool = False,  # False
        no_dec: bool = False,  # False
        act_ckpt: bool = False,  # True
        presence_head: bool = False,  # False
        dot_product_scorer=None,  # None
        cross_attend_prompt: "MultiheadAttention" | None = None,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            upsampling_stages=upsampling_stages,
            use_encoder_inputs=True,  # <-- encoder hidden statesを使用
            aux_masks=aux_masks,
            no_dec=no_dec,
            pixel_decoder=pixel_decoder,
            act_ckpt=act_ckpt,
        )
        self.d_model = hidden_dim

        if dot_product_scorer is not None:
            assert presence_head, "Specifying a dot product scorer without a presence head is likely a mistake"

        self.presence_head = None
        if presence_head:
            self.presence_head = (
                dot_product_scorer if dot_product_scorer is not None else LinearPresenceHead(self.d_model)
            )

        self.cross_attend_prompt = cross_attend_prompt
        if self.cross_attend_prompt is not None:
            self.cross_attn_norm = nn.LayerNorm(self.d_model)

        # Semantic segmentation head
        self.semantic_seg_head = nn.Conv2d(
            in_channels=self.pixel_decoder.out_dim,
            out_channels=1,
            kernel_size=1,
        )
        # Instance segmentation head
        self.instance_seg_head = nn.Conv2d(
            in_channels=self.pixel_decoder.out_dim,
            out_channels=self.d_model,  # 256
            kernel_size=1,
        )

    def forward(
        self,
        backbone_feats: list[torch.Tensor],
        obj_queries: torch.Tensor,  # (1,100,256)
        image_ids: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,  # (5184,1,256)
        prompt: Optional[torch.Tensor] = None,
        prompt_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        assert encoder_hidden_states is not None
        # batch size
        bs: int = encoder_hidden_states.shape[1]

        # (Option) encoder_hidden_statesにpromptの情報をクロスアテンションで融合
        if self.cross_attend_prompt is not None:
            tgt2 = self.cross_attn_norm(encoder_hidden_states)
            tgt2 = self.cross_attend_prompt(
                query=tgt2,
                key=prompt,
                value=prompt,
                key_padding_mask=prompt_mask,
            )[0]
            encoder_hidden_states = tgt2 + encoder_hidden_states

        presence_logit = None
        if self.presence_head is not None:  # <-- 不使用っぽい
            pooled_enc = encoder_hidden_states.mean(0)
            presence_logit = (
                self.presence_head(
                    pooled_enc.view(1, bs, 1, self.d_model),
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                )
                .squeeze(0)
                .squeeze(1)
            )

        # 親クラスの_pixel_embedを呼び出し。この中でPixelDecoderが呼び出される。
        pixel_embed: torch.Tensor = self._embed_pixels(
            backbone_feats=backbone_feats,
            image_ids=image_ids,
            encoder_hidden_states=encoder_hidden_states,
        )  # (1,256,288,288)

        # インスタンスレベルの特徴マップを取得
        instance_embeds: torch.Tensor = self.instance_seg_head(pixel_embed)  # (1,256,288,288)

        if self.no_dec:
            mask_pred: torch.Tensor = self.mask_predictor(instance_embeds)
        elif self.aux_masks:
            mask_pred: torch.Tensor = self.mask_predictor(obj_queries, instance_embeds)
        else:  # <-- 多分これが使われてる。MaskPredictorでの処理
            mask_pred: torch.Tensor = self.mask_predictor(
                obj_queries[-1],
                instance_embeds,
            )
        return {
            "pred_masks": mask_pred,  # (1,num_queries,288,288)
            "semantic_seg": self.semantic_seg_head(pixel_embed),  # 256 -> 1
            "presence_logit": presence_logit,  # None
        }
