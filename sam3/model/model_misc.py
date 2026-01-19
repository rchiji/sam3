from __future__ import annotations


# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""Various utility models"""

import copy
import math
import weakref
from collections.abc import Iterator
from contextlib import AbstractContextManager
from enum import auto, Enum
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing_extensions import override

# 循環import防止のため、コメントアウト
# from sam3.model.encoder import TransformerEncoderFusion
# from sam3.model.decoder import TransformerDecoder


def inverse_sigmoid(x, eps=1e-3):
    """
    The inverse function for sigmoid activation function.
    Note: It might face numberical issues with fp16 small eps.

    sigmoid後の値xから、sigmoid前の値を計算する
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MultiheadAttentionWrapper(nn.MultiheadAttention):
    """
    nn.MHAのneed_weightsをFalseに固定しただけ
    """

    def forward(self, *args, **kwargs):
        kwargs["need_weights"] = False
        return super().forward(*args, **kwargs)


class DotProductScoring(torch.nn.Module):
    def __init__(
        self,
        d_model,  # 256
        d_proj,  # 256
        prompt_mlp: MLP | None = None,
        clamp_logits=True,
        clamp_max_val=12.0,
    ):
        """

        model_builder.py内での定義例:
        def _create_dot_product_scoring() -> DotProductScoring:
            prompt_mlp = MLP(
                input_dim=256,
                hidden_dim=2048,
                output_dim=256,
                num_layers=2,
                dropout=0.1,
                residual=True,
                out_norm=nn.LayerNorm(256),
            )
            return DotProductScoring(d_model=256, d_proj=256, prompt_mlp=prompt_mlp)
        """
        super().__init__()
        self.d_proj: int = d_proj  # 256
        assert isinstance(prompt_mlp, torch.nn.Module) or prompt_mlp is None

        self.prompt_mlp: MLP | None = prompt_mlp  # an optional MLP projection for prompt
        self.prompt_proj: torch.nn.Linear = torch.nn.Linear(d_model, d_proj)  # 256 -> 256
        self.hs_proj: torch.nn.Linear = torch.nn.Linear(d_model, d_proj)  # 256 -> 256

        self.scale: float = float(1.0 / np.sqrt(d_proj))
        self.clamp_logits: bool = clamp_logits
        if self.clamp_logits:
            self.clamp_max_val: float = clamp_max_val

    def mean_pool_text(
        self,
        prompt: torch.Tensor,  # Prompt encoding (seq,bs,d_model) Ex: (34,1,256)
        prompt_mask: torch.Tensor,  # Prompt encoding mask (bs,seq) Ex: (1,34)
    ) -> torch.Tensor:
        """
        有効トークンで平均値を計算する

        Returns:
            pooled_prompt: Tensor  (bs,d_model)
        """

        # 1. 有効なトークン位置を示すマスクを作成
        # is_valid has shape (seq, bs, 1), where 1 is valid and 0 is padding
        is_valid: torch.Tensor = (
            (~prompt_mask).float().permute(1, 0)[..., None]
        )  # (bs, seq) -> (seq, bs) -> (seq, bs, 1)

        # 2. 有効なトークンの数を計算
        # num_valid has shape (bs, 1)
        num_valid: torch.Tensor = torch.clamp(torch.sum(is_valid, dim=0), min=1.0)

        # 3. 有効なトークンの特徴量を平均化
        # mean pool over all the valid tokens -- pooled_prompt has shape (bs, proj_dim)
        pooled_prompt: torch.Tensor = (prompt * is_valid).sum(dim=0) / num_valid
        return pooled_prompt

    def forward(
        self,
        hs: torch.Tensor,  # (num_layers,bs,num_queries,d_model)
        prompt: torch.Tensor,  # (seq, bs, d_model) Ex: (34,1,256)
        prompt_mask: torch.Tensor,  # (bs, seq) Ex: (1,34)
    ) -> torch.Tensor:
        """
        TransformerのDecoder層の出力特徴量と、プロンプト特徴量の類似度を計算する。

        hs: TransformerDecoderLayerの各層の出力特徴量をstackしたもの (num_layer, bs, num_query, d_model)
        prompt: Prompt encoding (seq, bs, d_model)
        prompt_mask: Mask for prompt encoding, where 1 is valid and 0 is padding, shape (bs, seq)

        Returns:
            scores: Tensor  (num_layer, bs, num_query, 1)
        """

        assert hs.dim() == 4 and prompt.dim() == 3 and prompt_mask.dim() == 2

        # apply MLP on prompt if specified
        if self.prompt_mlp is not None:
            # MLPで特徴量変換
            prompt: torch.Tensor = self.prompt_mlp(prompt)  # (seq,bs, d_model)

        # first, get the mean-pooled version of the prompt
        pooled_prompt: torch.Tensor = self.mean_pool_text(prompt, prompt_mask)  # (bs, d_model)

        # then, project pooled_prompt and hs to d_proj dimensions
        # nn.Linearでd_proj次元に変換
        proj_pooled_prompt: torch.Tensor = self.prompt_proj(pooled_prompt)  # (bs, d_proj)

        # nn.Linearでd_proj次元に変換
        proj_hs: torch.Tensor = self.hs_proj(hs)  # (num_layer, bs, num_query, d_proj)

        # finally, get dot-product scores of shape (num_layer, bs, num_query, 1)
        scores: torch.Tensor = torch.matmul(
            proj_hs,  # (num_layer, bs, num_query, d_proj)
            proj_pooled_prompt.unsqueeze(-1),  # (bs, d_proj, 1)
        )  # -> (num_layer, bs, num_query, 1)

        scores *= self.scale  # float(1.0 / np.sqrt(d_proj))

        # clamp scores to a max value to avoid numerical issues in loss or matcher
        if self.clamp_logits:
            scores.clamp_(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class TransformerWrapper(nn.Module):

    def __init__(
        self,
        encoder: "TransformerEncoderFusion",
        decoder: "TransformerDecoder",
        d_model: int,  # 256
        two_stage_type="none",  # ["none"] only for now
        pos_enc_at_input_dec=True,
    ):
        """
        Transformerのencoder, decoderをまとめたwrapperモデル
        forwardは無しで、encoder, decoderを個別に呼び出す形で使用する。
        """
        super().__init__()

        self.encoder: "TransformerEncoderFusion" = encoder
        self.decoder: "TransformerDecoder" = decoder
        self.num_queries: int | None = decoder.num_queries if decoder is not None else None  # 100 * batch_size
        self.pos_enc_at_input_dec: bool = pos_enc_at_input_dec

        # for two stage
        assert two_stage_type in ["none"], "unknown param {} of two_stage_type".format(two_stage_type)
        self.two_stage_type: str = two_stage_type

        self._reset_parameters()
        self.d_model: int = d_model

    def _reset_parameters(self):
        for n, p in self.named_parameters():
            if p.dim() > 1:
                if "box_embed" not in n and "query_embed" not in n and "reference_points" not in n:
                    nn.init.xavier_uniform_(p)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        dropout: float = 0.0,
        residual: bool = False,
        out_norm: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # whether to add the output as a residual connection to the input
        if residual and input_dim != output_dim:
            raise ValueError("residual is only supported if input_dim == output_dim")
        self.residual = residual
        # whether to apply a normalization layer to the output
        assert isinstance(out_norm, nn.Module) or out_norm is None
        self.out_norm = out_norm or nn.Identity()

    def forward(self, x):
        """
        ( nn.Linear -> ReLU -> Dropout ... ) x num_layers -> (optional residual) -> (optional norm)
        """
        orig_x = x
        for i, layer in enumerate(self.layers):
            x = self.drop(F.relu(layer(x))) if i < self.num_layers - 1 else layer(x)
        if self.residual:
            x = x + orig_x
        x = self.out_norm(x)
        return x


def get_clones(module, N):
    """
    moduleをN個複製してnn.ModuleListとして返す
    参考:
        Transformer layerを複数重ねるときに使う
    """
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_clones_seq(module, N):
    return nn.Sequential(*[copy.deepcopy(module) for i in range(N)])


def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_activation_module(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nn.ReLU
    if activation == "gelu":
        return nn.GELU
    if activation == "glu":
        return nn.GLU
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_valid_ratio(mask):
    """
    Calculate the valid ratio of non-masked elements in height and width dimensions of the mask.
    """
    _, H, W = mask.shape
    valid_H = torch.sum(~mask[:, :, 0], 1)
    valid_W = torch.sum(~mask[:, 0, :], 1)
    valid_ratio_h = valid_H.float() / H
    valid_ratio_w = valid_W.float() / W
    valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
    return valid_ratio


def gen_sineembed_for_position(
    pos_tensor: Tensor,  # (n_query,bs,4)
    num_feats: int = 256,
) -> Tensor:
    """
    sinusoidal position embeddingの2D版を生成

    (n_query,bs,4)の最後の軸は(cx,cy,w,h)のbbox座標を想定

    Returns:
        pos: Tensor  (n_query, bs, num_feats or num_feats*2)
    """
    assert num_feats % 2 == 0
    num_feats: int = num_feats // 2  # 128
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)

    scale: float = 2 * math.pi  # 位置情報を[0, 2pi]の範囲にスケーリング
    # 周波数成分の計算
    dim_t: Tensor = torch.arange(
        num_feats,
        dtype=torch.float32,
        device=pos_tensor.device,
    )
    dim_t: Tensor = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / num_feats)

    # 位置情報のスケーリング
    x_embed: Tensor = pos_tensor[:, :, 0] * scale  # 正規化x座標
    y_embed: Tensor = pos_tensor[:, :, 1] * scale  # 正規化y座標
    pos_x: Tensor = x_embed[:, :, None] / dim_t  # 周波数で割る
    pos_y: Tensor = y_embed[:, :, None] / dim_t  # 周波数で割る

    # sin, cosを交互に並べる
    pos_x: Tensor = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(
        2
    )  # (n_query, bs, 128)
    pos_y: Tensor = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(
        2
    )  # (n_query, bs, 128)

    # 最終的な位置埋め込みの生成
    if pos_tensor.size(-1) == 2:
        pos: Tensor = torch.cat((pos_y, pos_x), dim=2)  # (n_query, bs, 256)
    elif pos_tensor.size(-1) == 4:  # bbox座標(cx,cy,w,h)の場合
        # w成分の位置埋め込み
        w_embed: Tensor = pos_tensor[:, :, 2] * scale
        pos_w: Tensor = w_embed[:, :, None] / dim_t
        pos_w: Tensor = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(
            2
        )  # (n_query, bs, 128)

        # h成分の位置埋め込み
        h_embed: Tensor = pos_tensor[:, :, 3] * scale
        pos_h: Tensor = h_embed[:, :, None] / dim_t
        pos_h: Tensor = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(
            2
        )  # (n_query, bs, 128)

        # y, x, w, hの位置埋め込みを連結
        pos: Tensor = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)  # (n_query, bs, 512)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class SAM3Output(list):
    """
    A class representing the output of a SAM3 model.
    It provides an iterable interface that supports different iteration modes, including iterating over all steps per stage,
    last step per stage, and flattened output.
    Attributes:
        output: The output of the SAM3 model, represented as a list of lists.
        iter_mode: The current iteration mode.
    Example:
        >>> output = [[1, 2], [3, 4], [5, 6]]
        >>> sam3_output = SAM3Output(output)
        >>> for step in sam3_output:
        ...     print(step)
        [1, 2]
        [3, 4]
        [5, 6]
        >>> with SAM3Output.iteration_mode(SAM3Output.IterMode.LAST_STEP_PER_STAGE) as sam3_last_step_out:
        ...     for step in sam3_last_step_out:
        ...         print(step)
        [2]
        [4]
        [6]
        >>> with SAM3Output.iteration_mode(SAM3Output.IterMode.FLATTENED) as sam3_flattened_out:
        ...     for step in sam3_flattened_out:
        ...         print(step)
        1
        2
        3
        4
        5
        6
    """

    class IterMode(Enum):
        # Defines the type of iterator over ouptuts.
        ALL_STEPS_PER_STAGE = auto()
        LAST_STEP_PER_STAGE = auto()
        FLATTENED = (
            auto()
        )  # Returns each interactivity step as if it is a separate stage (this is used in SAM3Image model)

    def __init__(
        self,
        output: List[List[Dict]] = None,
        iter_mode: IterMode = IterMode.ALL_STEPS_PER_STAGE,
        loss_stages: Optional[List[int]] = None,
    ):
        if output is not None:
            assert (
                isinstance(output, list) and len(output) > 0 and isinstance(output[0], list)
            ), "Expected output to be a list of lists"
            self.output = output
        else:
            self.output = []
        assert isinstance(
            iter_mode, SAM3Output.IterMode
        ), f"iter_mode shoulf be of enum type 'SAM3Output.IterMode'. Got {type(iter_mode)}"

        self.iter_mode = iter_mode
        # We create a weak reference to self to be used in the lambda functions.
        # This is to avoid cyclic references and let SAM3Output be garabge collected.
        self_ref = weakref.ref(self)
        self._mode2iter = {
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE: lambda: iter(self_ref().output),
            SAM3Output.IterMode.LAST_STEP_PER_STAGE: lambda: (inner_list[-1] for inner_list in self_ref().output),
            SAM3Output.IterMode.FLATTENED: lambda: (
                element for inner_list in self_ref().output for element in inner_list
            ),
        }
        self.loss_stages = loss_stages

    @override
    def __iter__(self) -> Iterator:
        return self._mode2iter[self.iter_mode]()

    def __getitem__(self, index):
        """
        Returns the item at the specified index.
        Args:
            index (int): The index of the item to return.
        Returns:
            list or element: The item at the specified index.
        """
        assert isinstance(index, int), f"index should be an integer. Got {type(index)}"
        if self.iter_mode == SAM3Output.IterMode.ALL_STEPS_PER_STAGE:
            return self.output[index]
        elif self.iter_mode == SAM3Output.IterMode.LAST_STEP_PER_STAGE:
            return self.output[index][-1]
        elif self.iter_mode == SAM3Output.IterMode.FLATTENED:
            if index == -1:
                return self.self.output[-1][-1]
            else:
                flattened_output = sum(self.output, [])
                return flattened_output[index]

    class _IterationMode(AbstractContextManager):
        """
        A context manager that temporarily changes the iteration mode of a SAM3Output object.
        This class is used internally by the SAM3Output.iteration_mode method.
        """

        def __init__(self, model_output: "SAM3Output", iter_mode: "SAM3Output.IterMode"):
            self._model_output = model_output
            self._orig_iter_mode = model_output.iter_mode
            self._new_iter_mode = iter_mode

        @override
        def __enter__(self) -> "SAM3Output":
            self._model_output.iter_mode = self._new_iter_mode
            return self._model_output

        @override
        def __exit__(self, exc_type, exc_value, traceback):
            self._model_output.iter_mode = self._orig_iter_mode
            return super().__exit__(exc_type, exc_value, traceback)

    @staticmethod
    def iteration_mode(model_output: "SAM3Output", iter_mode: IterMode) -> _IterationMode:
        """
        Returns a context manager that allows you to temporarily change the iteration mode of the SAM3Output object.
        Args:
            model_output: The SAM3Output object.
            iter_mode: The new iteration mode.
        Returns:
            SAM3Output._IterationMode: A context manager that changes the iteration mode of the SAM3Output object.
        """
        return SAM3Output._IterationMode(model_output=model_output, iter_mode=iter_mode)

    def append(self, item: list):
        assert isinstance(item, list), f"Only list items are supported. Got {type(item)}"
        self.output.append(item)

    def __repr__(self):
        return self.output.__repr__()

    def __len__(self):
        if self.iter_mode in [
            SAM3Output.IterMode.ALL_STEPS_PER_STAGE,
            SAM3Output.IterMode.LAST_STEP_PER_STAGE,
        ]:
            return len(self.output)
        elif self.iter_mode == SAM3Output.IterMode.FLATTENED:
            flattened_output = sum(self.output, [])
            return len(flattened_output)
