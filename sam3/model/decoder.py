from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe
"""
Transformer decoder.
Inspired from Pytorch's version, adds the pre-norm variant
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sam3.sam.transformer import RoPEAttention
from torch import nn, Tensor
from torchvision.ops.roi_align import RoIAlign

from .act_ckpt_utils import activation_ckpt_wrapper
from .box_ops import box_cxcywh_to_xyxy
from .model_misc import (
    gen_sineembed_for_position,
    get_activation_fn,
    get_clones,
    inverse_sigmoid,
    MLP,
)


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        activation: str,  # "relu"
        d_model: int,  # 256
        dim_feedforward: int,  # 2048
        dropout: float,  # 0.1
        cross_attention: nn.Module | "MultiheadAttention",
        n_heads: int,  # 8
        use_text_cross_attention: bool = False,  # True
    ):
        super().__init__()

        # ---- Prompt encodingとのcross attention ----
        self.cross_attn: "MultiheadAttention" = cross_attention
        self.dropout1: nn.Dropout | nn.Identity = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)

        # ---- cross attention text ----
        self.use_text_cross_attention: bool = use_text_cross_attention
        if use_text_cross_attention:
            self.ca_text: nn.MultiheadAttention = nn.MultiheadAttention(
                d_model,
                n_heads,
                dropout=dropout,
            )
            self.catext_dropout: nn.Dropout | nn.Identity = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.catext_norm: nn.LayerNorm = nn.LayerNorm(d_model)

        # ---- self attention ----
        self.self_attn: nn.MultiheadAttention = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
        )
        self.dropout2: nn.Dropout | nn.Identity = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)

        # ---- ffn ----
        self.linear1: nn.Linear = nn.Linear(d_model, dim_feedforward)
        # F.relu | F.gelu | F.glu
        self.activation: callable = get_activation_fn(activation)
        self.dropout3: nn.Dropout | nn.Identity = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2: nn.Linear = nn.Linear(dim_feedforward, d_model)
        self.dropout4: nn.Dropout | nn.Identity = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3: nn.LayerNorm = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Tensor | None) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt: Tensor) -> Tensor:
        """
        Feed forward network
        nn.Linear -> Activation -> Dropout -> nn.Linear -> Dropout -> add to input -> LayerNorm
        """
        with torch.amp.autocast(device_type="cuda", enabled=False):
            tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        # for tgt
        tgt: Tensor,  # (num_query,bs=1,d_model=256) nn.Embeddingのweightsが初期値
        tgt_query_pos: (
            Tensor | None
        ) = None,  # pos for query. MLP(Sine(pos)) 2D sinusoidal pos embeddingをMLPに通したもの (num_query,bs,256)
        tgt_query_sine_embed: (
            Tensor | None
        ) = None,  # pos for query. Sine(pos) 2D sinusoidal pos embedding (num_query,bs,256)
        tgt_key_padding_mask: Tensor | None = None,
        tgt_reference_points: (
            Tensor | None
        ) = None,  # (num_query,bs=1,4) 参照点（TransformerDecoder内の処理で毎layerで更新されていくけど、不使用やん）
        memory_text: Tensor | None = None,  # (num_token, bs, d_model) Prompt encodingの出力 Ex: (34,1,256)
        text_attention_mask: Tensor | None = None,  # (bs, num_token) Prompt encoding mask Ex: (1,34)
        # for memory
        memory: Tensor | None = None,  # (hw, bs, d_model) [5184, 1, 256]　encoder_out["encoder_hidden_states"]
        memory_key_padding_mask: Tensor | None = None,
        memory_level_start_index: Tensor | None = None,  # (num_levels,) # flattenした時の、それぞれのlevelの開始index
        memory_spatial_shapes: Tensor | None = None,  # (bs, num_levels, 2) # 各levelのh,w情報 (1,1,2)[72,72]
        memory_pos: Tensor | None = None,  # pos for memory
        # self attention  # 画像位置埋め込みにlevel埋め込みを加算したもの [5184,1,256]
        self_attn_mask: Tensor | None = None,  # mask used for self-attention. None
        cross_attn_mask: Tensor | None = None,  # mask used for cross-attention. not None
        # dac
        dac: bool = False,  # True
        dac_use_selfatt_ln: bool = True,
        presence_token: Tensor | None = None,  # presence token (1,1,256) or None
        # skip inside deformable attn
        identity: float = 0.0,
        **kwargs,  # additional kwargs for compatibility
    ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -

        memoryとついているのはTransformerEncoderLayerの出力

        DAC（divide-and-conquer）：dac=True だと 前半のクエリにだけself-attn をかける。（Training時のみ有効）
        - tgt_o2o（前半）だけ更新   o2oは "one-to-one" の意味
            one-to-one 用のクエリ群（1クエリが基本1個の対象に対応する想定）
        - tgt_o2m（後半）はそのまま o2mは "one-to-many" の意味
            one-to-many 用のクエリ群（1つの条件/クエリが複数候補を拾う用途）
        - 最後に concat して戻す
        → 計算削減や、「役割の違うクエリ群」を混ぜすぎない目的。

        presence_token: Tensor | None = None
            CLSトークンのように先頭に1つだけ付与するトークン。boxが存在するかどうかの予測に使う。
        """
        # self attention
        if self.self_attn is not None:
            ## 1. queryを分割（dac=Trueのときのみ）
            if dac:
                # we only apply self attention to the first half of the queries
                assert tgt.shape[0] % 2 == 0
                num_o2o_queries: int = tgt.shape[0] // 2  # 100
                # qyery_embed weightの前半
                tgt_o2o: Tensor = tgt[:num_o2o_queries]
                tgt_query_pos_o2o: Tensor | None = tgt_query_pos[:num_o2o_queries]
                # qyery_embed weightの後半
                tgt_o2m: Tensor = tgt[num_o2o_queries:]
            else:
                tgt_o2o: Tensor = tgt
                tgt_query_pos_o2o: Tensor | None = tgt_query_pos

            ## 2. presence_tokenを先頭に追加
            if presence_token is not None:
                # presence_tokenを先頭に追加
                tgt_o2o: Tensor = torch.cat([presence_token, tgt_o2o], dim=0)  # (1+100, bs, d_model)
                # presence_token用のpos embをゼロ埋めで追加
                tgt_query_pos_o2o: Tensor = torch.cat([torch.zeros_like(presence_token), tgt_query_pos_o2o], dim=0)
                # presence_token用のsine embをゼロ埋めで追加
                tgt_query_pos = torch.cat([torch.zeros_like(presence_token), tgt_query_pos], dim=0)

            ## 3. Self Attention
            # tgt_o2oに位置埋め込みを加算したものをquery, keyに使う
            q: Tensor  # tgt_o2o + tgt_query_pos_o2o
            k: Tensor  # tgt_o2o + tgt_query_pos_o2o
            q = k = self.with_pos_embed(tgt_o2o, tgt_query_pos_o2o)
            # self attention
            tgt2: Tensor = self.self_attn(
                q,
                k,
                tgt_o2o,
                attn_mask=self_attn_mask,
            )[
                0
            ]  # <- nn.MHAの返り値がtupleなので[0]でTensorを取り出す (1+100, bs, d_model)
            tgt_o2o: Tensor = tgt_o2o + self.dropout2(tgt2)
            if dac:
                if not dac_use_selfatt_ln:
                    # LayerNorm
                    tgt_o2o: Tensor = self.norm2(tgt_o2o)
                # tgt_o2o（self attention後）と tgt_o2m（未処理）をconcat
                tgt: Tensor = torch.cat((tgt_o2o, tgt_o2m), dim=0)  # Recombine
                if dac_use_selfatt_ln:
                    # LayerNorm
                    tgt: Tensor = self.norm2(tgt)
            else:
                # DACでない場合はそのまま LayerNorm
                tgt: Tensor = tgt_o2o
                # LayerNorm
                tgt: Tensor = self.norm2(tgt)

        # 4. Prompt encodingとのcross attention
        if self.use_text_cross_attention:
            # cross attention
            tgt2: Tensor = self.ca_text(
                query=self.with_pos_embed(tgt, tgt_query_pos),
                key=memory_text,
                value=memory_text,
                key_padding_mask=text_attention_mask,
            )[
                0
            ]  # <- nn.MHAの返り値がtupleなので[0]でTensorを取り出す
            # DropOut or Identity
            tgt: Tensor = tgt + self.catext_dropout(tgt2)
            # LayerNorm
            tgt: Tensor = self.catext_norm(tgt)

        if presence_token is not None:
            presence_token_mask: Tensor = torch.zeros_like(cross_attn_mask[:, :1, :])
            cross_attn_mask: Tensor = torch.cat([presence_token_mask, cross_attn_mask], dim=1)  # (bs*nheads, 1+nq, hw)

        # 5. encoder出力とのCross attention
        tgt2: Tensor = self.cross_attn(
            query=self.with_pos_embed(tgt, tgt_query_pos),
            key=self.with_pos_embed(memory, memory_pos),  # endoer出力に位置埋め込み(+ level位置埋め込み)を加算したもの
            value=memory,  # encoder出力
            attn_mask=cross_attn_mask,
            key_padding_mask=(memory_key_padding_mask.transpose(0, 1) if memory_key_padding_mask is not None else None),
        )[
            0
        ]  # <- nn.MHAの返り値がtupleなので[0]でTensorを取り出す。(1+nq, bs, hw)

        tgt: Tensor = tgt + self.dropout1(tgt2)
        tgt: Tensor = self.norm1(tgt)

        # 6. ffn
        tgt: Tensor = self.forward_ffn(tgt)

        # 7. presence_tokenを分離して出力
        presence_token_out: Tensor | None = None
        if presence_token is not None:  # <-- True
            # presence_tokenを分離して出力
            presence_token_out: Tensor = tgt[:1]  # (1, bs, d_model)
            tgt: Tensor = tgt[1:]  # (num_query, bs, d_model)

        return tgt, presence_token_out


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,  # 256
        frozen: bool,  # False
        interaction_layer: nn.Module | None,  # None
        layer: TransformerDecoderLayer,
        num_layers: int,  # 6
        num_queries: int,  # 200
        return_intermediate: bool,  # True
        box_refine: bool = False,  # True
        num_o2m_queries: int = 0,  # 0
        dac: bool = False,  # True
        boxRPB: str = "none",  # "log"
        # Experimental: An object query for SAM 2 tasks
        instance_query: bool = False,
        # Defines the number of additional instance queries,
        # 1 or 4 are the most likely for single vs multi mask support
        num_instances: int = 1,  # Irrelevant if instance_query is False
        dac_use_selfatt_ln: bool = True,  # True
        use_act_checkpoint: bool = False,  # True
        compile_mode=None,
        presence_token: bool = False,  # True
        clamp_presence_logits: bool = True,
        clamp_presence_logit_max_val: float = 10.0,
        use_normed_output_consistently: bool = True,
        separate_box_head_instance: bool = False,
        separate_norm_instance: bool = False,
        resolution: int | None = None,  # 1008
        stride: int | None = None,  # 14
    ):
        """
        RPB は “Relative Position Bias”（相対位置バイアス）
        """
        super().__init__()
        self.d_model: int = d_model  # 256
        # TransformerDecoderLayerをnum_layers個copyしてModuleListに
        self.layers: torch.nn.ModuleList = get_clones(
            layer,
            num_layers,
        )
        self.fine_layers: list[nn.Module | None] = (
            get_clones(interaction_layer, num_layers) if interaction_layer is not None else [None] * num_layers
        )
        self.num_layers: int = num_layers  # 6
        self.num_queries: int = num_queries  # 200
        self.dac: bool = dac  # True
        if dac:
            self.num_o2m_queries: int = num_queries  # 200
            tot_num_queries: int = num_queries
        else:
            self.num_o2m_queries: int = num_o2m_queries
            tot_num_queries: int = num_queries + num_o2m_queries

        self.norm: nn.LayerNorm = nn.LayerNorm(d_model)
        self.return_intermediate: bool = return_intermediate

        # bbox prediction head (nn.Linear->ReLU->Dropout)*num_layers
        # 参照boxとの差分を予測するMLP
        self.bbox_embed: MLP = MLP(
            input_dim=d_model,
            hidden_dim=d_model,
            output_dim=4,  # (cx,cy,w,h)用の4次元
            num_layers=3,
        )  # (256 dim -> 4 dim)

        # query indexを256次元に変換するembedding。このweightをdecoderの初期値として使用するだけで、query_embedは訓練されない。
        self.query_embed: nn.Embedding = nn.Embedding(
            tot_num_queries,
            d_model,
        )  # 200 -> 256
        self.instance_query_embed = None
        self.instance_query_reference_points = None
        self.use_instance_query: bool = instance_query  # False
        self.num_instances: int = num_instances  # 1 <-- use_instance_queryがTrueのときのみ意味がある
        self.use_normed_output_consistently: bool = use_normed_output_consistently  # True

        self.instance_norm = nn.LayerNorm(d_model) if separate_norm_instance else None
        self.instance_bbox_embed = None
        if separate_box_head_instance:  # Falseなので不使用
            self.instance_bbox_embed = MLP(
                input_dim=d_model,
                hidden_dim=d_model,
                output_dim=4,
                num_layers=3,
            )
        if instance_query:  # Falseなので不使用
            self.instance_query_embed = nn.Embedding(num_instances, d_model)

        self.box_refine: bool = box_refine  # True
        if box_refine:
            # MLPの最後の層のweightとbiasを0初期化
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

            # 参照点の埋め込み
            self.reference_points: nn.Embedding = nn.Embedding(num_queries, 4)  # 200 -> 4
            if instance_query:  # Falseなので不使用
                self.instance_reference_points: nn.Embedding = nn.Embedding(num_instances, 4)

        assert boxRPB in ["none", "log", "linear", "both"]
        self.boxRPB: str = boxRPB  # "log"
        if boxRPB != "none":
            try:
                nheads: int = self.layers[0].cross_attn_image.num_heads  # 8
            except AttributeError:
                nheads: int = self.layers[0].cross_attn.num_heads  # 8

            # xとy座標を別々にエンコードするMLP
            n_input: int = 4 if boxRPB == "both" else 2  # 2
            self.boxRPB_embed_x: MLP = MLP(
                input_dim=n_input,  # 2
                hidden_dim=d_model,  # 256
                output_dim=nheads,  # 8
                num_layers=2,
            )
            self.boxRPB_embed_y: MLP = MLP(
                input_dim=n_input,  # 2
                hidden_dim=d_model,  # 256
                output_dim=nheads,  # 8
                num_layers=2,
            )
            self.compilable_cord_cache: tuple[torch.Tensor, torch.Tensor] | None = None
            self.compilable_stored_size: tuple[int, int] | None = None
            self.coord_cache: dict = {}

            # 事前に解像度情報から正規化座標を計算してキャッシュしておく
            if resolution is not None and stride is not None:
                feat_size: int = resolution // stride  # 1008 // 14 = 72
                cache_device: str = "cuda" if torch.cuda.is_available() else "cpu"
                # 0-71の72個の要素を72で割って0-1に正規化した座標を取得
                coords_h, coords_w = self._get_coords(
                    feat_size,  # 72
                    feat_size,  # 72
                    device=cache_device,
                )
                self.compilable_cord_cache: tuple[torch.Tensor, torch.Tensor] = (coords_h, coords_w)
                self.compilable_stored_size: tuple[int, int] = (feat_size, feat_size)

        # torchvision.ops.roi_align.RoIAlignインスタンス <-- 不使用
        self.roi_pooler: RoIAlign = (
            RoIAlign(
                output_size=7,
                spatial_scale=1,
                sampling_ratio=-1,
                aligned=True,
            )
            if interaction_layer is not None
            else None
        )
        if frozen:  # False
            for p in self.parameters():
                p.requires_grad_(False)

        # logitsのclamp設定
        self.clamp_presence_logits: bool = clamp_presence_logits  # True
        self.clamp_presence_logit_max_val: float = clamp_presence_logit_max_val  # 10.0

        self.presence_token: nn.Embedding | None = None
        if presence_token:  # True
            ## すでに存在しているかを示す presence token を導入
            # presence token用の埋め込み。1 -> 256次元に変換するnn.Embedding
            self.presence_token: nn.Embedding = nn.Embedding(1, d_model)
            # presence tokenの有無を予測するhead
            self.presence_token_head = MLP(
                input_dim=d_model,  # 256
                hidden_dim=d_model,  # 256
                output_dim=1,
                num_layers=3,
            )
            # presence token用のLayerNorm
            self.presence_token_out_norm: nn.LayerNorm = nn.LayerNorm(d_model)

        # sinusoidal positional embeddingを256次元に変換するMLP
        self.ref_point_head = MLP(
            input_dim=2 * self.d_model,  # 512
            hidden_dim=self.d_model,  # 256
            output_dim=self.d_model,  # 256
            num_layers=2,
        )
        self.dac_use_selfatt_ln: bool = dac_use_selfatt_ln  # True
        self.use_act_checkpoint: bool = use_act_checkpoint  # True

        # query_embedのweightを正規分布で初期化
        nn.init.normal_(self.query_embed.weight.data)
        if self.instance_query_embed is not None:  # 不使用
            nn.init.normal_(self.instance_query_embed.weight.data)

        assert self.roi_pooler is None
        assert self.return_intermediate, "support return_intermediate only"
        assert self.box_refine, "support box refine only"

        self.compile_mode: None = compile_mode
        self.compiled: bool = False
        # We defer compilation till after the first forward, to first warm-up the boxRPB cache

        # assign layer index to each layer so that some layers can decide what to do based on which layer index they are (e.g. cross attention to memory bank only in selected layers)
        for layer_idx, layer in enumerate(self.layers):
            # TransformerDecoderLayerにlayer_idx属性を追加
            layer.layer_idx = layer_idx

    @staticmethod
    def _get_coords(H, W, device) -> tuple[torch.Tensor, torch.Tensor]:
        """
        0からH-1までのH個の要素をHで割って0-1に正規化した座標を取得
        0からW-1までのW個の要素をWで割って0-1に正規化した座標を取得
        """
        coords_h = torch.arange(0, H, device=device, dtype=torch.float32) / H
        coords_w = torch.arange(0, W, device=device, dtype=torch.float32) / W
        return coords_h, coords_w

    def _get_rpb_matrix(
        self,
        reference_boxes: torch.Tensor,
        feat_size: tuple[int, int],
    ) -> torch.Tensor:
        """
        1. reference_boxes（各queryの参照box）を xyxy に変換
        2. feature map の正規化座標 coords_w, coords_h（0〜1）を作る
        3. 各 query ごとに
            deltas_x = coords_w - [xmin,xmax]
            deltas_y = coords_h - [ymin,ymax]
            を作る（「各ピクセル位置が、そのboxの左右上下の境界からどれだけ離れてるか」）
        4. それを log 変換したり（boxRPB="log"）、生+log両方使ったり（both）
        5. boxRPB_embed_x / boxRPB_embed_y（MLP）で head数（n_heads）次元に落とす
        6. deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2) で (H,W) を合成して
        7. 最終的に B: [bs, n_heads, num_queries, H*W] を作る

        この関数の出力を memory_mask として cross-attn に渡す（※名前は mask だけど実体は “bias”）
        つまり boxRPB は **「参照boxに基づく2D相対位置バイアス」**をクロスアテンションに入れて、「このqueryは、このbox近辺（や内側）を見やすくする」みたいな誘導をしてます。
        """
        H, W = feat_size  # 72, 72
        boxes_xyxy: torch.Tensor = box_cxcywh_to_xyxy(reference_boxes).transpose(0, 1)  # (1,200,4)

        bs: int  # 1
        num_queries: int  # 200
        bs, num_queries, _ = boxes_xyxy.shape
        self.compilable_cord_cache: tuple[
            torch.Tensor, torch.Tensor
        ]  # # 0-71の72個の要素を72で割って0-1に正規化した座標を取得
        if (
            self.compilable_cord_cache is None
            or self.compilable_stored_size != (H, W)
            or any(coord.device != reference_boxes.device for coord in self.compilable_cord_cache)
        ):
            self.compilable_cord_cache: tuple[torch.Tensor, torch.Tensor] = self._get_coords(
                H,
                W,
                reference_boxes.device,
            )
            self.compilable_stored_size: tuple[int, int] = (H, W)

        if torch.compiler.is_dynamo_compiling() or self.compilable_stored_size == (
            H,
            W,
        ):
            # good, hitting the cache, will be compilable
            coords_h, coords_w = self.compilable_cord_cache
        else:
            # cache miss, will create compilation issue
            # In case we're not compiling, we'll still rely on the dict-based cache
            if feat_size not in self.coord_cache:
                # このfeature size (72)での正規化座標(mesh grid)をcache
                self.coord_cache[feat_size] = self._get_coords(H, W, reference_boxes.device)
            coords_h, coords_w = self.coord_cache[feat_size]

            assert coords_h.shape == (H,)
            assert coords_w.shape == (W,)

        # 正規化y座標(0-1)と参照ボックスのymin,ymaxとの差分を計算
        deltas_y: torch.Tensor = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]  # (1,200,2)
        deltas_y = deltas_y.view(bs, num_queries, -1, 2)  # (1,200,72,2)
        # 正規化x座標(0-1)と参照ボックスのxmin,xmaxとの差分を計算
        deltas_x: torch.Tensor = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]  # (1,200,2)
        deltas_x: torch.Tensor = deltas_x.view(bs, num_queries, -1, 2)  # (1,200,72,2)

        if self.boxRPB in ["log", "both"]:
            # 相対位置（Δx, Δy）を “符号付きログ圧縮” して、近距離は細かく・遠距離は粗く扱える特徴量に変換
            # 1. 8倍してlog2 -> 0から離れているほど値が大きく出やすい
            # 2. signで正or負を取り出し 1 or -1
            # 3. log2(8)で割れば大体元のscale (-1~1あたり)に戻る

            deltas_x_log: torch.Tensor = deltas_x * 8  # normalize to -8, 8
            deltas_x_log: torch.Tensor = (
                torch.sign(deltas_x_log) * torch.log2(torch.abs(deltas_x_log) + 1.0) / np.log2(8)
            )

            deltas_y_log: torch.Tensor = deltas_y * 8  # normalize to -8, 8
            deltas_y_log: torch.Tensor = (
                torch.sign(deltas_y_log) * torch.log2(torch.abs(deltas_y_log) + 1.0) / np.log2(8)
            )

            if self.boxRPB == "log":  # <-- これを採用
                deltas_x: torch.Tensor = deltas_x_log
                deltas_y: torch.Tensor = deltas_y_log
            else:
                deltas_x: torch.Tensor = torch.cat([deltas_x, deltas_x_log], dim=-1)
                deltas_y: torch.Tensor = torch.cat([deltas_y, deltas_y_log], dim=-1)

        if self.training:
            assert self.use_act_checkpoint, "activation ckpt not enabled in decoder"

        # xとy座標それぞれに対してMLPを適用
        deltas_x: torch.Tensor = activation_ckpt_wrapper(self.boxRPB_embed_x)(
            x=deltas_x,  # (bs=1,num_queries=200,W=72,2)
            act_ckpt_enable=self.training and self.use_act_checkpoint,
        )  # (bs=1,num_queries=200,W=72,n_heads=8)
        deltas_y: torch.Tensor = activation_ckpt_wrapper(self.boxRPB_embed_y)(
            x=deltas_y,  # (bs=1,num_queries=200,H=72,2)
            act_ckpt_enable=self.training and self.use_act_checkpoint,
        )  # (bs=1,num_queries=200,H=72,n_heads=8)
        if not torch.compiler.is_dynamo_compiling():
            assert deltas_x.shape[:3] == (bs, num_queries, W)
            assert deltas_y.shape[:3] == (bs, num_queries, H)

        # deltas_Xとdeltas_Yを合成 broadcastで足し算
        B: torch.Tensor = deltas_y.unsqueeze(3) + deltas_x.unsqueeze(2)  # (bs=1,num_queries=200,H=72,W=72,n_heads=8)

        if not torch.compiler.is_dynamo_compiling():
            assert B.shape[:4] == (bs, num_queries, H, W)

        B: torch.Tensor = B.flatten(2, 3)  # (bs=1,num_queries=200,H*W=5184,n_heads=8)
        B: torch.Tensor = B.permute(0, 3, 1, 2)  # (bs=1,n_heads=8, num_queries=200, H*W=5184)
        B: torch.Tensor = B.contiguous()  # memeff attn likes ordered strides
        if not torch.compiler.is_dynamo_compiling():
            assert B.shape[2:] == (num_queries, H * W)
        return B

    def forward(
        self,
        tgt,  # nn.Embedding.weight (200,1,256)
        memory,  # TransformerEncoderの出力。encoder_hidden_state [5184,1,256]
        tgt_mask: Tensor | None = None,  # None
        memory_mask: Tensor | None = None,  # None
        tgt_key_padding_mask: Tensor | None = None,  # None
        memory_key_padding_mask: Tensor | None = None,  # TransformerEncoderの出力。"padding_mask"。結局None
        pos: (
            Tensor | None
        ) = None,  # TransformerEncoderの出力。画像位置埋め込みにlevel埋め込みを加算したもの [5184,1,256]
        reference_boxes: Tensor | None = None,  # None (num_queries,bs,4)
        # for memory
        level_start_index: Tensor | None = None,  # TransformerEncoderの出力。"level_start_index" num_levels=1
        spatial_shapes: (
            Tensor | None
        ) = None,  # TransformerEncoderの出力。"spatial_shapes" (bs=1,num_levels=1,2) (72,72)
        valid_ratios: Tensor | None = None,  # TransformerEncoderの出力。"valid_ratios"
        # for text
        memory_text: Tensor | None = None,  # Prompt encoding Ex: (34,1,256) not None
        text_attention_mask: Tensor | None = None,  # Prompt encoding mask Ex: (1,34) not None
        # if `apply_dac` is None, it will default to `self.dac`
        apply_dac: bool | None = None,  # True
        is_instance_prompt: bool = False,
        decoder_extra_kwargs: dict | None = None,
        # ROI memory bank
        obj_roi_memory_feat=None,
        obj_roi_memory_mask=None,
        box_head_trk=None,
    ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: \\sum{hw}, bs, d_model
            - pos: \\sum{hw}, bs, d_model
            - reference_boxes: nq, bs, 4 (after sigmoid)
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        if memory_mask is not None:
            assert (
                self.boxRPB == "none"
            ), "inputting a memory_mask in the presence of boxRPB is unexpected/not implemented"

        # model_builder.pyではdac=True
        apply_dac: bool = apply_dac if apply_dac is not None else self.dac
        if apply_dac:
            assert (tgt.shape[0] == self.num_queries) or (
                self.use_instance_query and (tgt.shape[0] == self.instance_query_embed.num_embeddings)
            )

            tgt: Tensor = tgt.repeat(2, 1, 1)  # (400,1,256)
            # note that we don't tile tgt_mask, since DAC doesn't
            # use self-attention in o2m queries
            if reference_boxes is not None:
                assert (reference_boxes.shape[0] == self.num_queries) or (
                    self.use_instance_query and (reference_boxes.shape[0] == self.instance_query_embed.num_embeddings)
                )
                reference_boxes = reference_boxes.repeat(2, 1, 1)

        bs: int = tgt.shape[1]  # batch size=1
        intermediate: list[Tensor] = []
        intermediate_presence_logits: list[Tensor] = []
        presence_feats: Tensor | None = None

        if self.box_refine:  # <-- True
            # reference_boxesがNoneなら初期化
            # nn.Embedding (200->4) の weightをsigmoidで0-1に変換したものを参照点として使う
            if reference_boxes is None:  # <-- True
                # In this case, we're in a one-stage model, so we generate the reference boxes
                # nn.Embedding (200->4) の weight を参照点の初期値として使う
                reference_boxes: Tensor = self.reference_points.weight.unsqueeze(1)  # (200,1,4)
                # DAC時は2倍に拡張 (400,1,4) <-- 推論では不使用
                reference_boxes: Tensor = (
                    reference_boxes.repeat(2, bs, 1) if apply_dac else reference_boxes.repeat(1, bs, 1)
                )
                # sigmoidで0-1に変換
                reference_boxes: Tensor = reference_boxes.sigmoid()
            intermediate_ref_boxes: list[Tensor] = [reference_boxes]
        else:
            reference_boxes: Tensor | None = None
            intermediate_ref_boxes: list[Tensor] | None = None

        output: Tensor = tgt
        presence_out: Tensor | None = None
        if self.presence_token is not None and is_instance_prompt is False:  # <-- True

            # nn.Embedding (1->256) の weight を presence token として使う
            # expand to batch dim
            presence_out: Tensor = self.presence_token.weight[None].expand(1, bs, -1)  # (1,1,256)

        box_head: MLP = self.bbox_embed
        if is_instance_prompt and self.instance_bbox_embed is not None:  # <-- Falseなので不使用
            # box_headを instance_bbox_embed に切り替え
            box_head = self.instance_bbox_embed

        out_norm: nn.LayerNorm = self.norm
        if is_instance_prompt and self.instance_norm is not None:  # <-- Falseなので不使用
            # out_normを instance_norm に切り替え
            out_norm = self.instance_norm

        # TransformerDecoderLayerを順番に適用
        for layer_idx, layer in enumerate(self.layers):
            ## 1. 現在の候補box位置を基に参照点を計算
            # reference_boxes[:, :, None] -> (200,1,1,4)
            # valid_ratios -> (bs=1,nlevel=1,2)
            # torch.cat([valid_ratios, valid_ratios], -1)[None, :] -> (1,1,4)
            reference_points_input: Tensor = (
                reference_boxes[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            )  # (nq=200,bs=1,nlevel=1,4)

            ## 2. 現在の候補参照点を基に位置埋め込みを計算
            # Sinusoidal positional embeddingの2D版を生成 cx,cy,w,h各々に対して128次元ずつで合計512次元
            query_sine_embed: Tensor = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :], self.d_model
            )  # (nq,bs,d_model*2 =512)

            ## 3. 位置埋め込みを次元削減しつつ、trainableな埋め込みに変換
            # conditional query
            # Sinusoidal positional embeddingをMLPで256次元に変換
            query_pos: Tensor = self.ref_point_head(query_sine_embed)  # (nq=200,bs=1,d_model=256)

            if self.boxRPB != "none" and reference_boxes is not None:  # <-- True
                assert spatial_shapes.shape[0] == 1, "only single scale support implemented"

                ## 4. boxRPB。候補boxの相対位置バイアスをMLPしてcross attentionのattn_mask作成。headごとにweightを変えたもの。
                memory_mask: Tensor = self._get_rpb_matrix(
                    reference_boxes,
                    (spatial_shapes[0, 0], spatial_shapes[0, 1]),  # (72,72)
                )  # (bs=1, n_heads=8, nq=200, H*W=5184)
                memory_mask: Tensor = memory_mask.flatten(0, 1)  # (bs*n_heads=8, nq=200, H*W=5184)

            if self.training:
                assert self.use_act_checkpoint, "Activation checkpointing not enabled in the decoder"

            ## 5. TransformerDecoderLayerのforwardを呼び出し
            output: torch.Tensor  # (200,1,256)
            presence_out: torch.Tensor  # (1,1,256)
            output, presence_out = activation_ckpt_wrapper(layer)(
                tgt=output,  # nn.Embedding.weight (200,1,256) or 前のlayerのoutput
                tgt_query_pos=query_pos,  # query_pos (200,1,256)
                tgt_query_sine_embed=query_sine_embed,  # (200,1,512) # <-- 不使用
                tgt_key_padding_mask=tgt_key_padding_mask,  # None
                tgt_reference_points=reference_points_input,  # 現在の候補参照点 (200,1,1,4) ※ 毎layerで更新したものを渡しているけど、layerのfowardには不使用。
                memory_text=memory_text,  # Prompt encoding Ex: (34,1,256) not None
                text_attention_mask=text_attention_mask,  # Prompt encoding mask Ex: (1,34) not None
                memory=memory,  # TransformerEncoderの出力。encoder_hidden_state [5184,1,256]
                memory_key_padding_mask=memory_key_padding_mask,  # None
                memory_level_start_index=level_start_index,  # TransformerEncoderの出力。"level_start_index" num_levels=1
                memory_spatial_shapes=spatial_shapes,  # TransformerEncoderの出力。"spatial_shapes" (bs=1,num_levels=1,2)
                memory_pos=pos,  # 画像位置埋め込みにlevel埋め込みを加算したもの
                self_attn_mask=tgt_mask,  # None
                cross_attn_mask=memory_mask,  # (bs*n_heads=8, nq=200, H*W=5184)
                dac=apply_dac,  # True
                dac_use_selfatt_ln=self.dac_use_selfatt_ln,  # True
                presence_token=presence_out,  # 前のlayerのpresence token (1,1,256)
                **(decoder_extra_kwargs or {}),
                act_ckpt_enable=self.training and self.use_act_checkpoint,
                # ROI memory bank
                obj_roi_memory_feat=obj_roi_memory_feat,  # None
                obj_roi_memory_mask=obj_roi_memory_mask,  # None
            )

            ## 6. 候補box位置の更新
            # iter update
            if self.box_refine:  # <-- True
                # sigmoid前の値に戻す
                reference_before_sigmoid: Tensor = inverse_sigmoid(reference_boxes)  # (200,1,4)

                if box_head_trk is None:  # <-- True
                    # delta_unsig = self.bbox_embed(output)
                    if not self.use_normed_output_consistently:  # <-- True
                        # MLPでboxのdeltaを予測
                        delta_unsig: Tensor = box_head(output)
                    else:
                        delta_unsig: Tensor = box_head(out_norm(output))
                else:
                    # box_head_trk use a separate box head for tracking queries
                    Q_det = decoder_extra_kwargs["Q_det"]
                    assert output.size(0) >= Q_det
                    delta_unsig_det = self.bbox_embed(output[:Q_det])
                    delta_unsig_trk = box_head_trk(output[Q_det:])
                    delta_unsig = torch.cat([delta_unsig_det, delta_unsig_trk], dim=0)

                # 候補box位置に差分を加算して更新
                outputs_unsig: Tensor = delta_unsig + reference_before_sigmoid
                # sigmoidで0-1に変換して新しい参照点を得る
                new_reference_points: Tensor = outputs_unsig.sigmoid()

                reference_boxes: Tensor = new_reference_points.detach()  # (200,1,4)
                if layer_idx != self.num_layers - 1:
                    intermediate_ref_boxes.append(new_reference_points)
            else:
                raise NotImplementedError("not implemented yet")

            ## 7. intermediateにこのlayerのoutputを保存
            intermediate.append(out_norm(output))

            ## 8. presence tokenの処理
            if self.presence_token is not None and is_instance_prompt is False:  # <-- True

                # norm, mlp head
                intermediate_layer_presence_logits = self.presence_token_head(
                    self.presence_token_out_norm(presence_out)
                ).squeeze(
                    -1
                )  # (1,1)

                # clamp to mitigate numerical issues
                if self.clamp_presence_logits:  # <-- True 10.0以内にclamp
                    intermediate_layer_presence_logits.clamp(
                        min=-self.clamp_presence_logit_max_val,
                        max=self.clamp_presence_logit_max_val,
                    )

                # save per-layer presence logits
                intermediate_presence_logits.append(intermediate_layer_presence_logits)

                # cloneしてるけど、次のlayerで使用されているわけではない。
                # 最後のlayerのpresence_outを返すためだけに使われているっぽい。
                presence_feats: Tensor = presence_out.clone()

        if not self.compiled and self.compile_mode is not None:
            self.forward = torch.compile(self.forward, mode=self.compile_mode, fullgraph=True)
            self.compiled = True

        return (
            torch.stack(intermediate),
            torch.stack(intermediate_ref_boxes),
            (
                torch.stack(intermediate_presence_logits)
                if self.presence_token is not None and is_instance_prompt is False
                else None
            ),
            presence_feats,
        )


class TransformerEncoderCrossAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        frozen: bool,
        pos_enc_at_input: bool,
        layer,
        num_layers: int,
        use_act_checkpoint: bool = False,
        batch_first: bool = False,  # Do layers expect batch first input?
        # which layers to exclude cross attention? default: None, means all
        # layers use cross attention
        remove_cross_attention_layers: Optional[list] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = get_clones(layer, num_layers)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.use_act_checkpoint = use_act_checkpoint

        if frozen:
            for p in self.parameters():
                p.requires_grad_(False)

        self.batch_first = batch_first

        # remove cross attention layers if specified
        self.remove_cross_attention_layers = [False] * self.num_layers
        if remove_cross_attention_layers is not None:
            for i in remove_cross_attention_layers:
                self.remove_cross_attention_layers[i] = True
        assert len(self.remove_cross_attention_layers) == len(self.layers)

        for i, remove_cross_attention in enumerate(self.remove_cross_attention_layers):
            if remove_cross_attention:
                self.layers[i].cross_attn_image = None
                self.layers[i].norm2 = None
                self.layers[i].dropout2 = None

    def forward(
        self,
        src,  # self-attention inputs
        prompt,  # cross-attention inputs
        src_mask: Optional[Tensor] = None,  # att.mask for self-attention inputs
        prompt_mask: Optional[Tensor] = None,  # att.mask for cross-attention inputs
        src_key_padding_mask: Optional[Tensor] = None,
        prompt_key_padding_mask: Optional[Tensor] = None,
        src_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        prompt_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        feat_sizes: Optional[list] = None,
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        if isinstance(src, list):
            assert isinstance(src_key_padding_mask, list) and isinstance(src_pos, list)
            assert len(src) == len(src_key_padding_mask) == len(src_pos) == 1
            src, src_key_padding_mask, src_pos = (
                src[0],
                src_key_padding_mask[0],
                src_pos[0],
            )

        assert src.shape[1] == prompt.shape[1], "Batch size must be the same for src and prompt"

        output = src

        if self.pos_enc_at_input and src_pos is not None:
            output = output + 0.1 * src_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)
            prompt = prompt.transpose(0, 1)
            prompt_pos = prompt_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = activation_ckpt_wrapper(layer)(
                tgt=output,
                memory=prompt,
                tgt_mask=src_mask,
                memory_mask=prompt_mask,
                tgt_key_padding_mask=src_key_padding_mask,
                memory_key_padding_mask=prompt_key_padding_mask,
                pos=prompt_pos,
                query_pos=src_pos,
                dac=False,
                attn_bias=None,
                act_ckpt_enable=self.training and self.use_act_checkpoint,
                **kwds,
            )
            normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)

        return {
            "memory": normed_output,
            "pos_embed": src_pos,
            "padding_mask": src_key_padding_mask,
        }


class TransformerDecoderLayerv1(nn.Module):
    def __init__(
        self,
        activation: str,
        cross_attention: nn.Module,
        d_model: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        self_attention: nn.Module,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attention
        self.cross_attn_image = cross_attention

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation_str = activation
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm

        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        **kwargs,
    ):
        q = k = tgt + query_pos if self.pos_enc_at_attn else tgt

        # Self attention
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # Cross attention to image
        tgt2 = self.cross_attn_image(
            query=tgt + query_pos if self.pos_enc_at_cross_attn_queries else tgt,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        dac: bool = False,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        **kwargs,
    ):
        if dac:
            # we only apply self attention to the first half of the queries
            assert tgt.shape[0] % 2 == 0
            other_tgt = tgt[tgt.shape[0] // 2 :]
            tgt = tgt[: tgt.shape[0] // 2]
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(
            q,
            k,
            value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        if dac:
            # Recombine
            tgt = torch.cat((tgt, other_tgt), dim=0)
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            query=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            key=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
            attn_bias=attn_bias,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        dac: bool = False,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        **kwds: Any,
    ) -> torch.Tensor:
        fwd_fn = self.forward_pre if self.pre_norm else self.forward_post
        return fwd_fn(
            tgt,
            memory,
            dac=dac,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            pos=pos,
            query_pos=query_pos,
            attn_bias=attn_bias,
            **kwds,
        )


class TransformerDecoderLayerv2(TransformerDecoderLayerv1):
    def __init__(self, cross_attention_first=False, *args: Any, **kwds: Any):
        super().__init__(*args, **kwds)
        self.cross_attention_first = cross_attention_first

    def _forward_sa(self, tgt, query_pos):
        # Self-Attention
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        if self.cross_attn_image is None:
            return tgt

        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        dac: bool,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        attn_bias: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ):
        assert dac is False
        assert tgt_mask is None
        assert memory_mask is None
        assert tgt_key_padding_mask is None
        assert memory_key_padding_mask is None
        assert attn_bias is None

        if self.cross_attention_first:
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
            tgt = self._forward_sa(tgt, query_pos)
        else:
            tgt = self._forward_sa(tgt, query_pos)
            tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)

        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, *args: Any, **kwds: Any) -> torch.Tensor:
        if self.pre_norm:
            return self.forward_pre(*args, **kwds)
        raise NotImplementedError
