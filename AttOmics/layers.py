import logging

import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from networkx.algorithms import bipartite
from scipy import sparse
from torch import Tensor
from torch import nn as nn

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class LinearBNDropout(nn.Module):
    def __init__(self, input_dim, output_dim, batch_norm=False, dropout=0):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        else:
            self.batch_norm = nn.Identity()

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(self.batch_norm(x))
        return self.dropout(x)


class SparseLinearMLP(nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_group: int,
        group_size: int,
        connectivity: Tensor,
        proj_dim: list,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        """_summary_

        Args:
            in_features (int): _description_
            out_features (int): _description_
            n_group (int): _description_
            group_size (int): _description_
            connectivity (Tensor): _description_
            proj_dim (list): _description_ list of list 1st level correspond to group, second level to proj dim, one element per proj
            bias (bool, optional): _description_. Defaults to True.
            device (_type_, optional): _description_. Defaults to None.
            dtype (_type_, optional): _description_. Defaults to None.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias


        in_feat_conn = sum(map(len, connectivity))
        assert n_group == len(connectivity)
        assert in_feat_conn == in_features
        assert out_features == group_size * n_group
        index_per_grp_in = connectivity
        index_per_grp_out = [
            out_ for out_ in torch.split(torch.arange(0, out_features), group_size)
        ]
        self.n_grp = n_group
        logger.debug(f"SparseLinearMLP: {self.n_grp} group(s)")
        n_params = sum(
            [
                len(in_) * len(out_)
                for in_, out_ in zip(index_per_grp_in, index_per_grp_out)
            ]
        )
        self.sparsity = 1 - n_params / (in_features * out_features)
        layer_dim = [
            [idx_grp_in.size(0)] + grp_proj_dim
            for idx_grp_in, grp_proj_dim in zip(index_per_grp_in, proj_dim)
        ]
        logger.debug(layer_dim)
        self.list_linear = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        nn.Linear(grp_dim[i], grp_dim[i + 1], bias=bias)
                        for i in range(0, len(grp_dim) - 1)
                    ]
                )
                for grp_dim in layer_dim
            ]
        )
        logger.debug(f"Found {self.n_grp} groups from the connectivity input")
        logger.debug(f"Layers for group MLP: {self.list_linear}")
        for i in range(self.n_grp):
            self.register_buffer(f"index_group_in_{i}", index_per_grp_in[i])
            self.register_buffer(f"index_group_out_{i}", index_per_grp_out[i])

    def index_group_i(self, i, prefix="index_group_in"):
        return self.__getattr__(f"{prefix}_{i}")

    def index_groups(self, prefix="index_group_in"):
        for i in range(self.n_grp):
            yield self.__getattr__(f"{prefix}_{i}")

    def forward(self, x):
        return torch.cat(
            [
                module(x[:, idx_lst])
                for idx_lst, module in zip(self.index_groups(), self.list_linear)
            ],
            1,
        )

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, sparsity={self.sparsity}"


class GroupInteraction(nn.Module):
    def __init__(
        self,
        group_size: int,
        norm_layer: nn.Module,
        num_heads: int = 1,
        residual_connection: bool = True,
        attn_dropout: float = 0,
    ):
        """Implements the MultiheadAttention + Add&Norm operation from the transformer architecture

        Args:
            group_size (int): Size of each group, use to set the attention layer dimension
            norm_layer (nn.Module): An instanciated `Pytorch Module` used to perform normalisation
            at the end of the module.
            residual_connection (bool, optional): Wether to include or not a residual connection after
            the attention mechanism as in the paper `Attention is all you need`. Defaults to True.
        """
        super().__init__()
        assert (group_size % num_heads) == 0
        self.attention = nn.MultiheadAttention(
            embed_dim=group_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=attn_dropout,
        )
        self.norm_layer = norm_layer
        self.residual = residual_connection

    def forward(self, x: Tensor) -> Tensor:
        z, att_w = self.attention(
            query=x, key=x, value=x, need_weights=True, average_attn_weights=False
        )
        # possible dropout after SA
        if self.residual:
            z = z + x
        return self.norm_layer(z), att_w


class AttOmicsLayer(nn.Module):
    def __init__(
        self,
        in_features,
        grouped_dim,
        n_group,
        group_size,
        connectivity,
        norm_layer,
        num_heads,
        group_proj_dim,  # list of list, one for each group
        residual_connection=True,
        attn_dropout: float = 0,
    ) -> None:
        super().__init__()
        assert grouped_dim == group_size * n_group
        assert len(group_proj_dim) == n_group
        assert all([len(hidden_dim_grp) >= 1 for hidden_dim_grp in group_proj_dim])
        self.n_group = n_group
        self.group_size = group_size
        # Transform each group with a MLP
        self.grouped_mlp = GroupedMLP(
            in_features, grouped_dim, n_group, group_size, connectivity, group_proj_dim
        )
        # Apply attention on group sequence
        self.interaction = GroupInteraction(
            group_size=group_size,
            norm_layer=norm_layer,
            residual_connection=residual_connection,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        # N: BAtch size, G: number of groups, s: size of a group
        # Input dim: Nxd (d: number of input features )
        logger.debug(f"AtomicsLayer input shape: {x.shape}")
        x = self.grouped_mlp(x)  # dim: Nx(G*s)
        logger.debug(f"Shape after Group MLP: {x.shape}")
        x = x.view(-1, self.n_group, self.group_size)  # dim: NxGxs
        logger.debug(f"Reshape to match attention input: {x.shape}")
        x, att_w = self.interaction(x)  # dim: NxGxs
        return x.view(-1, self.n_group * self.group_size), att_w  # dim: Nx(G*s)


GroupedMLP = SparseLinearMLP
