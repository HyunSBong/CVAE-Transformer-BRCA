import sys
from pathlib import Path

path_root = Path(__file__)
sys.path.append(str(path_root))


import json
import logging
import math
import time
from typing import List, Tuple, Union

import joblib
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from k_means_constrained import KMeansConstrained
from networkx.readwrite import json_graph
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn

from base import BaseModelSingleOmics
from layers import AttOmicsLayer, LinearBNDropout

assets_path = Path(__file__).resolve().parents[1] / "assets"
logger = logging.getLogger(__name__)

cache_dir = Path(__file__).parents[2] / "cache"
if not cache_dir.exists():
    cache_dir.mkdir()
connectivity_cache = joblib.Memory(cache_dir, mmap_mode="c", verbose=0)


class SetDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        if "genes" in dct:
            dct.update({"genes": set(dct["genes"])})
        if "gene_final" in dct:
            dct.update({"gene_final": set(dct["gene_final"])})
        return dct


@connectivity_cache.cache
def constrained_kmeans_fun(data, n_clusters):
    in_dim = data.shape[0]
    size_min = math.floor(in_dim / n_clusters)
    scaler = StandardScaler().fit(data)
    X = scaler.transform(data)
    clf = KMeansConstrained(
        n_clusters=n_clusters, size_min=size_min, size_max=None, n_jobs=4
    )
    clf = clf.fit(X)
    return clf.labels_


def constrained_kmeans_grouping(
    in_features: int,
    proj_size: int,
    n_group: int,
    train_data: DataFrame,
    **kwargs,
) -> Tuple:
    assert in_features == train_data.shape[1]
    name_to_pos = {name: pos for pos, name in enumerate(train_data.columns)}
    logger.info("Clustering the features with Constrained k-means")
    t = time.time()
    labels_ = constrained_kmeans_fun(train_data.transpose(), n_group)
    logger.info(f"Clustering took {time.time() - t} seconds")
    idx_in = (
        pd.DataFrame.from_dict(
            {"group": labels_, "feature": train_data.columns.map(name_to_pos)}
        )
        .groupby("group")["feature"]
        .apply(list)
    )
    group_name = [f"Cluster {i}" for i in range(len(idx_in))]

    return (
        [torch.tensor(idx_in_) for idx_in_ in idx_in],
        group_name,
        [[proj_size] for _ in range(n_group)],
    )


@connectivity_cache.cache
def get_go(go_graph, group_size, size_threshold, strict):
    root = list(nx.topological_sort(go_graph))[0]
    selected_nodes = set()
    to_visit = list(go_graph.successors(root))
    do_not_visit = set()
    while len(to_visit) > 0:
        node = to_visit.pop()
        if strict and node in do_not_visit:
            continue
        size = go_graph.nodes[node].get("size", 0)
        if (
            abs(size - group_size) < size_threshold
        ):  # good node keep it, do not explore its successors
            selected_nodes.add(node)
            do_not_visit.update(list(nx.descendants(go_graph, node)))
            continue
        # the node has not a correct size continue to explore
        to_visit.extend(list(go_graph.successors(node)))
    return selected_nodes


def gene_ontology_grouping(
    in_features: int,
    proj_size: int,
    n_group: int,
    train_data: DataFrame,
    n_gene_per_group: int = 5000,
    threshold: int = 500,
    drop_remainder: bool = False,
    strict: bool = False,
    **kwargs,
) -> Tuple:
    logger.info(
        "When grouping features with GO, `n_group` parameters"
        + " is ignored. The number of group will depend on the `n_gene_per_group` and the"
        + " selected `threshold`."
    )
    name_to_pos = {
        name.split(".")[0]: pos for pos, name in enumerate(train_data.columns)
    }
    # get group size from out_features and n_group
    train_data.columns = [name.split(".")[0] for name in train_data.columns]
    with Path(__file__).with_name("go_graph_annotation.json").open("r") as f:
        go_graph_json = json.load(f, cls=SetDecoder)
    logger.debug("Loading Graph")
    go_graph = json_graph.node_link_graph(go_graph_json)
    selected_nodes = get_go(go_graph, n_gene_per_group, threshold, strict)

    genes_in = [go_graph.nodes[n]["gene_final"] for n in selected_nodes]
    mean_size = sum(map(len, genes_in)) / len(genes_in)
    unique_genes = set()
    for gene_lst in genes_in:
        unique_genes.update(gene_lst)
    idx_in = [[name_to_pos.get(gene) for gene in gene_list] for gene_list in genes_in]
    logger.info(
        f"Identified {len(selected_nodes)} nodes in the ontology to create groups."
    )
    n_genes = len(unique_genes)
    logger.info(f"The {len(selected_nodes)} selected nodes represent {n_genes} genes.")
    logger.debug(f"{selected_nodes}")
    group_name = selected_nodes
    if not drop_remainder:
        logger.info("Clustering features that did not map onto the ontology.")
        data_remainder = train_data.copy().drop(columns=unique_genes)
        n_group_remainder = math.floor(data_remainder.shape[1] / mean_size)
        random_conn = random_grouping(data_remainder.shape[1], 1000, n_group_remainder)
        idx_in_remainder, _ = zip(*random_conn)
        # map idx to original data
        # transform idx_in_remainder to name using data_remainder
        genes_in_remainder = [
            data_remainder.columns[idx_list.tolist()].to_list()
            for idx_list in idx_in_remainder
        ]
        idx_in += [
            [name_to_pos.get(gene) for gene in gene_list]
            for gene_list in genes_in_remainder
        ]
        group_name += [f"Random {i}" for i in range(len(idx_in))]
    n_group = len(idx_in)

    return (
        [torch.tensor(idx_in_) for idx_in_ in idx_in],
        group_name,
        [[proj_size] for _ in range(n_group)],
    )


def random_grouping(
    in_features: int,
    proj_size: int,
    n_group: int,
    train_data: DataFrame = None,
    matrix: bool = False,
    **kwargs,
) -> Union[Tensor, Tuple]:
    """_summary_

    Args:
        in_features (int): _description_
        out_features (int): _description_
        n_group (int): _description_
        gene_name (List[str], optional): _description_. Defaults to None.
        matrix (bool, optional): Defaults to False.

    Returns:
        torch.Tensor: _description_
    """

    idx_in = torch.randperm(in_features)
    chunk_sizes = (idx_in.size(0) // n_group) + (
        np.arange(n_group) < (idx_in.size(0) % n_group)
    )
    idx_in = idx_in.split(chunk_sizes.tolist(), dim=0)
    out_features = n_group * proj_size
    idx_out = [
        [out_]
        for out_ in torch.split(
            torch.arange(0, out_features), math.ceil(out_features / n_group)
        )
    ]
    if matrix:
        return torch.cat(
            [
                torch.stack(
                    torch.meshgrid(idx_in[i], idx_out[i], indexing="xy"), 0
                ).view(2, -1)
                for i in range(n_group)
            ],
            dim=1,
        ).view(2, -1)
    else:
        group_name = [f"Random {i}" for i in range(len(idx_in))]
        return idx_in, group_name, [[proj_size] for _ in range(n_group)]


def predefined_grouping(
    in_features: int,
    out_features: int,
    n_group: int,
    train_data: DataFrame = None,
    matrix: bool = False,
    **kwargs,
) -> Tuple:
    path = kwargs.get("path", None)
    if path is None:
        raise ValueError("No path to load the groups from")
    # load groups_in
    groups = json.load(path)
    genes_in = list(groups.values())
    group_name = list(groups.keys())
    # generate matching groups out
    idx_out = [
        [out_]
        for out_ in torch.split(
            torch.arange(0, out_features), math.ceil(out_features / n_group)
        )
    ]
    name_to_pos = {
        name.split(".")[0]: pos for pos, name in enumerate(train_data.columns)
    }

    unique_genes = set()
    for gene_lst in genes_in:
        unique_genes.update(gene_lst)
    idx_in = [[name_to_pos.get(gene) for gene in gene_list] for gene_list in genes_in]

    assert len(idx_in) == len(idx_out), "Non matching number of groups"
    return [
        (idx_in_, idx_out_) for idx_in_, idx_out_ in zip(idx_in, idx_out)
    ], group_name


def msigdb_hallmark(
    in_features: int,
    proj_size: int,
    n_group: int,
    train_data: DataFrame = None,
    **kwargs,
):
    name_to_pos = {
        name.split(".")[0]: pos for pos, name in enumerate(train_data.columns)
    }
    with open(assets_path / "hallmarks_genes.json", "r") as f:
        hallmarks = json.load(f)

    min_size = min(map(len, hallmarks.values()))
    max_size = 70
    group_name = []
    idx_in = []
    grp_proj_dim = []
    for hallmark_name, genes in hallmarks.items():
        genes_tensor = torch.tensor([name_to_pos.get(gene) for gene in genes])
        n_genes = len(genes)
        if n_genes > min_size:
            r1, r2 = min_size / max_size, min_size / n_genes
            if n_genes > max_size:
                n = round(math.log(r2) / math.log(r1))
            else:
                n = 1

            group_size = [round(n_genes * r1**i) for i in range(1, n)] + [min_size]
        group_name.append(hallmark_name.replace("HALLMARK_", ""))
        idx_in.append(genes_tensor)
        # idx_out.append([torch.arange(0, gsize) for gsize in group_size])
        grp_proj_dim.append(group_size)
    return idx_in, group_name, grp_proj_dim


def gene_ontology_slim_grouping(
    in_features: int,
    proj_size: int,
    n_group: int,
    train_data: DataFrame,
    drop_remainder: bool = True,
    strategy: str = "split",
    min_size: int = 200,
    max_size: int = 500,
    **kwargs,
) -> Tuple:
    assert (
        max_size > min_size
    ), "[GO Slim] `max_size` cannot be lower or equal than `min_size`"
    logger.info(
        "When grouping features with GO Slim, `n_group` or `group_size` parameters"
        + " are ignored. The number of group will depend on the `strategy` and the `group_size`"
        + f" will be `min_size`={min_size}"
    )
    name_to_pos = {
        name.split(".")[0]: pos for pos, name in enumerate(train_data.columns)
    }
    # path = "go_slims_group.json"
    with Path(__file__).with_name("go_slims_group.json").open("r") as f:
        slim_groups = json.load(f)

    if not drop_remainder:
        logger.warning("Using GO slims with unmapped genes is not yet supported")
        pass
    group_name = []
    idx_in = []
    n_groups = 0
    if strategy == "split":
        for go_term, genes in slim_groups.items():
            n_genes = len(genes)
            genes_tensor = torch.tensor([name_to_pos.get(gene) for gene in genes])
            if n_genes >= min_size:
                if n_genes >= max_size * 1.5:  # because of rounding strategy
                    split_groups = round(n_genes / max_size)
                    logger.info(
                        f"Go term {go_term} has been split in {split_groups} groups."
                    )
                    group_name.extend(
                        [f"{go_term}-{i}" for i in range(1, split_groups + 1)]
                    )

                    chunk_sizes = (n_genes // split_groups) + (
                        np.arange(split_groups) < (n_genes % split_groups)
                    )
                    genes_split = genes_tensor.split(chunk_sizes.tolist(), dim=0)
                    idx_in.extend(genes_split)
                    n_groups += split_groups
                else:
                    group_name.append(go_term)
                    idx_in.append(genes_tensor)
                    n_groups += 1
        grp_proj_dim = [[min_size] for _ in range(n_groups)]

    elif strategy == "project":
        # idx_out = []
        grp_proj_dim = []
        for go_term, genes in slim_groups.items():
            genes_tensor = torch.tensor([name_to_pos.get(gene) for gene in genes])
            n_genes = len(genes)
            if n_genes > min_size:
                r1, r2 = min_size / max_size, min_size / n_genes
                if n_genes > max_size:
                    n = round(math.log(r2) / math.log(r1))
                else:
                    n = 1

                group_size = [round(n_genes * r1**i) for i in range(1, n)] + [
                    min_size
                ]
                group_name.append(go_term)
                idx_in.append(genes_tensor)
                # idx_out.append([torch.arange(0, gsize) for gsize in group_size])
                grp_proj_dim.append(group_size)
                n_groups += 1

    else:
        ValueError(f"Unsupported strategy: {strategy}")
    return idx_in, group_name, grp_proj_dim


def create_mlp(input_dim, hidden_dim, num_classes, dropout=0, batch_norm=True):
    n_hidden = len(hidden_dim)
    model = []
    if n_hidden >= 1:
        model.append(
            LinearBNDropout(
                input_dim,
                hidden_dim[0],
                batch_norm=False,
                dropout=0,
            )
        )

        for i in range(n_hidden - 1):
            model.append(
                LinearBNDropout(
                    hidden_dim[i],
                    hidden_dim[i + 1],
                    batch_norm=batch_norm,
                    dropout=dropout,
                )
            )
        model.append(LinearBNDropout(hidden_dim[-1], num_classes, batch_norm, dropout))
    else:
        model.append(LinearBNDropout(input_dim, num_classes, batch_norm, dropout))
    return nn.Sequential(*model)


GeneGroupCreation = {
    "random": random_grouping,
    "clustering": constrained_kmeans_grouping,
    "gene_ontology": gene_ontology_grouping,
    "gene_ontology_slim": gene_ontology_slim_grouping,
    "msigdb_hallmark": msigdb_hallmark,
}
HeadGeneration = {"mlp": create_mlp}


def get_group_size(input_dim, n_layers, target_dim, n_group):
    a = (input_dim - target_dim) / (-n_layers)
    b = (input_dim + target_dim) / 2 - a / 2 * n_layers

    return ((a * np.arange(1, n_layers + 1) + b) // n_group).astype(int)


class AttOmics(BaseModelSingleOmics):
    def __init__(
        self,
        input_dim: dict,  # a list of dimension
        num_classes: int,
        n_group: int,
        attention_norm: str,
        grouping_method: str,
        optimizer_init: dict,
        scheduler_init: dict,
        label_type: str,
        n_layers: int = 1,
        num_heads: int = 1,
        group_size: Union[int, List[int]] = None,
        head_norm: str = None,
        sa_residual_connection: bool = True,
        head_residual_connection: bool = False,
        class_weights: np.array = None,
        head_dropout: float = 0,
        head_batch_norm: bool = True,
        reuse_grp: bool = False,
        constant_group_size: bool = True,
        head_input_dim: int = None,
        head_hidden_ratio: list = None,
        head_hidden_dim: list = None,
        head_n_layers=None,  # used for search to get a default number of layers, if None set by len(head_hidden_dim)
        train_data=None,
        connectivity=None,
        group_name=None,
        grp_proj_dim=None,
        attn_dropout: float = 0,
        connectivity_kwargs: Union[dict, str] = {},
    ):
        self.connectivity = connectivity
        self.group_name = group_name
        self.grp_proj_dim = grp_proj_dim
        if isinstance(connectivity_kwargs, str):
            logger.info(f"`connectivity_kwargs` string: {connectivity_kwargs}")
            try:
                connectivity_kwargs = json.loads(connectivity_kwargs)
                logger.info(f"`connectivity_kwargs` from string: {connectivity_kwargs}")
            except:
                logger.error("Cannot load `connectivity_kwargs`")
                logger.exception()
        self.save_hyperparameters(
            ignore=["optimizer_init", "scheduler_init", "train_data"]
        )
        logger.debug(self.hparams)
        if group_size is None and constant_group_size:
            raise ValueError("With constant group size, you must provide a group_size")

        self.group_creation_fn = GeneGroupCreation.get(grouping_method, None)
        if self.group_creation_fn is None:
            raise KeyError(f"Unsupported grouping method: {grouping_method}")
        if grouping_method == "msigdb_hallmark":
            # logger.info("Setting groups to 50.")
            n_group = 50
            group_size = 32
            head_input_dim = n_group * group_size

        self.head_generation_fn = create_mlp
        # Group size checks
        if constant_group_size and group_size is None:
            raise ValueError(
                "You must specify a `group_size` when creating a model with a constant group size across layers"
            )
        elif constant_group_size and isinstance(group_size, int):
            head_input_dim = group_size * n_group
            group_size = [group_size] * n_layers
        elif (
            not constant_group_size
            and not isinstance(group_size, list)
            and head_input_dim is None
        ):
            raise ValueError(
                "Cannot guess the `group_size`. You must at least specify the `head_input_dim` so that the group size can be computed."
            )
        elif (
            not constant_group_size
            and not isinstance(group_size, list)
            and not head_input_dim is None
        ):
            if isinstance(group_size, int):
                logger.warning(
                    "The specified group size will be ignored and the group size"
                    + "will be computed to linearly decrease the dimension across layer"
                    + "To force a specific group size pass a list as argument for `group_size`."
                )
            self.used_get_group_size = True
            group_size = get_group_size(
                input_dim, n_layers, head_input_dim, n_group
            ).tolist()
            logger.info(
                f"From the specified `head_input_dim`: {head_input_dim}, "
                + f"the follwing `group_size`: {group_size} has been computed to linearly decrease the size across blocks."
            )
        elif (
            isinstance(group_size, list)
            and len(group_size) == n_layers
            and group_size[-1] * n_group == head_input_dim
        ):
            # likely correspond to checkpoint restore
            pass
        else:  # should not reach
            # reach with checkpointing
            logger.error(
                f"Unexpected case: group_size={group_size}, "
                + f"constant_group_size={constant_group_size}, "
                + f"head_input_dim={head_input_dim}"
            )
            raise ValueError("Unknown case for `group_size`")

        assert (
            len(group_size) == n_layers
        ), f"There is {len(group_size)} group_size but you requested {n_layers} layers."
        group_size_set = set(group_size)
        if constant_group_size:
            assert (
                len(group_size_set) == 1
            ), f"You requested a constant group size across layers but found {len(group_size_set)} different group_size: {group_size_set}"
        logger.debug(f"Group size:  {group_size}")
        logger.debug(f"num_heads: {num_heads}")
        logger.debug(f"n_group: {n_group}")
        # handle num_heads > 1
        # group_size must be divisible by the number of heads
        # if num_heads > 1:
        #     group_size = [
        #         int(num_heads * math.ceil(g_size / num_heads)) for g_size in group_size
        #     ]
        group_size = self._update_group_size_num_heads(num_heads, group_size)
        assert all(
            [g_size % num_heads == 0 for g_size in group_size]
        ), f"One of the group_size {group_size} is not divisible by the number of heads {num_heads}"
        assert not any(
            [g_size == 0 for g_size in group_size]
        ), f"One of the group_size is 0: {group_size}"
        logger.debug(f"Group size corrected for num_heads:  {group_size}")

        grouped_dim = [g_size * n_group for g_size in group_size]
        if grouped_dim[-1] != head_input_dim:
            logger.warning(
                "After computing a group size compatible with the requested number of heads, "
                + "the dimension of the last `AttOmics` layer does not match the requested `head_input_dim`"
                + f"{head_input_dim} != {grouped_dim[-1]}"
                + "\n"
                + f"Updating `head_input_dim` to {grouped_dim[-1]}."
            )
            head_input_dim = grouped_dim[-1]
        logger.debug(f"Gouped dim: {grouped_dim}")

        if head_hidden_ratio is not None:
            if head_n_layers is not None and len(head_hidden_ratio) != head_n_layers:
                raise ValueError(
                    f"The number of ratio must match the number of layers in the head, {len(head_hidden_ratio)} != {head_n_layers}"
                )
        if head_hidden_dim is not None:
            if head_n_layers is not None and len(head_hidden_dim) != head_n_layers:
                raise ValueError(
                    f"The number of dim must match the number of layers in the head, {len(head_hidden_ratio)} != {head_n_layers}"
                )
        # handle hidden ratios
        head_hidden_dim = self._get_classif_head_dim(
            head_input_dim, head_hidden_dim, head_hidden_ratio
        )
        logger.debug(f"head_hidden_dim: {head_hidden_dim}")
        if head_n_layers is not None and head_n_layers != len(head_hidden_dim):
            raise ValueError(
                "The number of head_n_layers does not match the length of head_hidden_dim"
            )
        elif head_n_layers is None:
            head_n_layers = len(head_hidden_dim)
        self.head_n_layers = head_n_layers

        self.grouping_method = grouping_method
        self.group_size = group_size
        self.n_group = n_group
        self.grouped_dim = grouped_dim
        self.sa_residual_connection = sa_residual_connection
        self.head_residual_connection = head_residual_connection
        self.head_norm = head_norm
        self.head_dropout = head_dropout
        self.head_batch_norm = head_batch_norm
        self.head_hidden_dim = head_hidden_dim
        self.head_hidden_ratio = head_hidden_ratio
        self.num_heads = num_heads
        self.n_layers = n_layers
        self.reuse_grp = reuse_grp
        self.constant_group_size = constant_group_size
        self.head_input_dim = head_input_dim
        self.connectivity_kwargs = connectivity_kwargs

        super().__init__(
            optimizer_init=optimizer_init,
            scheduler_init=scheduler_init,
            label_type=label_type,
            num_classes=num_classes,
            class_weights=class_weights,
            train_data=train_data,
        )
        n_group = self.n_group
        grouped_dim = self.grouped_dim
        head_input_dim = self.head_input_dim
        group_size = self.group_size
        connectivity = self.connectivity
        group_name = self.group_name
        grp_proj_dim = self.grp_proj_dim
        head_hidden_ratio = None
        head_hidden_dim = self.head_hidden_dim
        self.save_hyperparameters(
            ignore=["optimizer_init", "scheduler_init", "train_data"]
        )
        logger.debug(self.hparams)

    def init_model(self):
        logger.info("Generating the initial connectivity matrix")
        if self.connectivity is not None:
            connectivity = self.connectivity
            group_name = self.group_name
            grp_proj_dim = self.grp_proj_dim
        else:
            connectivity, group_name, grp_proj_dim = self.group_creation_fn(
                self.hparams.input_dim,
                self.group_size[0],
                self.n_group,
                self.train_data,
                **self.connectivity_kwargs,
            )
            self.connectivity = connectivity
            self.group_name = group_name
            self.grp_proj_dim = grp_proj_dim
        n_group_conn = len(connectivity)
        input_dim_virtual = 0
        for idx_in_ in connectivity:
            input_dim_virtual += idx_in_.size(0)
        input_dim = input_dim_virtual
        if n_group_conn != self.n_group:
            logger.warning(
                f"The connectivity matrix changed the requested number of group from {self.n_group} to {n_group_conn}"
            )  # this should happen only when using go based connectivity

            self.n_group = n_group_conn
            logger.debug(f"Number of group: {self.n_group}")
            # TODO: Improve this part causing issues (especially with GO)
            if self.used_get_group_size:

                if self.grouping_method == "gene_ontology_slim":
                    grp_out_size = set([grp[-1] for grp in grp_proj_dim])
                    assert (
                        len(grp_out_size) == 1
                    ), f"You have different dimension outputs for your groups: {grp_out_size}"

                    # self.group_size = [
                    #     list(grp_out_size)[0] for _ in range(self.n_layers)
                    # ]
                    if self.n_layers > 1:
                        self.group_size = (
                            list(grp_out_size)
                            + get_group_size(
                                list(grp_out_size)[0] * self.n_group,
                                self.n_layers - 1,
                                self.head_input_dim,
                                self.n_group,
                            ).tolist()
                        )
                        logger.info(
                            f"Updating `group_size` to be compatible with ontology: {self.group_size}"
                        )
                    else:
                        self.group_size = list(grp_out_size)
                else:
                    self.group_size = get_group_size(
                        input_dim_virtual,
                        self.n_layers,
                        self.head_input_dim,
                        self.n_group,
                    ).tolist()
                self.group_size = self._update_group_size_num_heads(
                    self.num_heads, self.group_size
                )
            elif (
                self.constant_group_size
                and self.grouping_method == "gene_ontology_slim"
            ):
                grp_out_size = set([grp[-1] for grp in grp_proj_dim])
                self.group_size = [list(grp_out_size)[0] for _ in range(self.n_layers)]
            else:
                logger.warning(
                    "You manually specified the `group_size` for each layer."
                    + " But the grouping methods change the number of groups and potentially the input dimension."
                    + "If you know what you are doing, you can continue."
                )
            self.grouped_dim = [g_size * self.n_group for g_size in self.group_size]
            logger.debug(f"Grouped dim: {self.grouped_dim}")
            self.head_input_dim = self.grouped_dim[-1]

            if self.head_hidden_ratio is not None:
                self.head_hidden_dim = self._get_classif_head_dim(
                    self.head_input_dim, dim=None, ratio=self.head_hidden_ratio
                )

        # self.group_name = group_name
        if self.hparams.attention_norm == "batch_norm":
            norm_layer = nn.BatchNorm1d(self.n_group)
        elif self.hparams.attention_norm == "layer_norm":
            norm_layer = nn.LayerNorm(self.group_size[0])

        input_layer = AttOmicsLayer(
            in_features=input_dim,
            grouped_dim=self.grouped_dim[0],
            n_group=self.n_group,
            group_size=self.group_size[0],
            connectivity=connectivity,
            norm_layer=norm_layer,
            num_heads=self.num_heads,
            residual_connection=self.sa_residual_connection,
            group_proj_dim=grp_proj_dim,
            attn_dropout=self.hparams.attn_dropout,
        )
        attOmics_layers = [input_layer]
        for i in range(1, self.n_layers):
            if self.reuse_grp:
                connectivity = [
                    idx
                    for idx in attOmics_layers[i - 1].grouped_mlp.index_groups(
                        "index_group_out"
                    )
                ]
            else:
                # TODO: will probably fail, check not reuse_grp and random grouping only
                connectivity, _, _ = self.group_creation_fn(
                    self.grouped_dim[i - 1],
                    self.grouped_dim[i],
                    self.n_group,
                )

            if self.hparams.attention_norm == "batch_norm":
                norm_layer = nn.BatchNorm1d(self.n_group)
            elif self.hparams.attention_norm == "layer_norm":
                norm_layer = nn.LayerNorm(self.group_size[i])
            attOmics_layers.append(
                AttOmicsLayer(
                    in_features=self.grouped_dim[i - 1],
                    grouped_dim=self.grouped_dim[i],
                    n_group=self.n_group,
                    group_size=self.group_size[i],
                    connectivity=connectivity,
                    norm_layer=norm_layer,
                    num_heads=self.num_heads,
                    residual_connection=self.sa_residual_connection,
                    group_proj_dim=[[self.group_size[i]] for _ in range(self.n_group)],
                    attn_dropout=self.hparams.attn_dropout
                    # subsequent layer, GroupedMLP project in a single layer no multiple layer
                )
            )
        self.attOmics_layers = nn.ModuleList(attOmics_layers)
        # TODO: handle case head_hidden_dim is empty or only one element
        if len(self.head_hidden_dim) == 0:
            self.head = nn.Identity()
            norm_dim = self.head_input_dim
        else:
            self.head = self.head_generation_fn(
                input_dim=self.head_input_dim,
                hidden_dim=self.head_hidden_dim[:-1],
                num_classes=self.head_hidden_dim[-1],
                dropout=self.head_dropout,
                batch_norm=self.head_batch_norm,
            )  # must be a sequential model
            norm_dim = self.head_hidden_dim[-1]
        if not isinstance(self.head, (nn.Sequential, nn.Identity)):
            raise TypeError(
                f"head must be a sequential model but got: {type(self.head)}"
            )
        if self.head_norm is None:
            self.head_norm_layer = nn.Identity()
        elif self.head_norm == "batch_norm":
            self.head_norm_layer = nn.BatchNorm1d(norm_dim)
        elif self.head_norm == "layer_norm":
            self.head_norm_layer = nn.LayerNorm(norm_dim)
        self.out = nn.Linear(norm_dim, self.num_classes)

    def model(self, x, return_attention=False):
        logger.debug(f"DataDim: {x.shape}")
        attention_map = []
        for layer in self.attOmics_layers:
            x, att_w = layer(x)
            logger.debug(f"Dim after attention block: {x.shape}")
            attention_map.append(att_w)

        y = self.head(x)
        logger.debug(f"Dim after classification head: {y.shape}")
        if self.head_residual_connection:
            y = y + x
        y = self.head_norm_layer(y)
        logger.debug(f"Dim after norm head: {y.shape}")
        if return_attention:
            return self.out(y), torch.stack(attention_map).transpose(0, 1)
        else:
            return self.out(y)

    def test_step(self, batch, batch_idx, dataloader_idx=None):

        x = batch.get("x", None)
        y = batch.get("label")

        event = batch.get("event", None)
        y_hat, attention_map = self.model(x, True)

        if self.hparams.label_type == "survival":
            loss = self.loss_fn(y_hat, y, event)
        else:
            y_hat = F.log_softmax(y_hat, dim=1)
            loss = self.loss_fn(y_hat, y)

        return {
            "loss": loss,
            "preds": y_hat.detach(),
            "target": y,
            "event": event,
            "attention": attention_map.detach(),  # dim: N x n_layers x num_heads x n_group x n_group
        }

    def _update_group_size_num_heads(self, num_heads, group_size):
        if num_heads > 1:
            new_group_size = [
                int(num_heads * math.ceil(g_size / num_heads)) for g_size in group_size
            ]
            if new_group_size != group_size:
                logger.info(
                    f"Updating `group_size` to be divisible by `num_heads` ({num_heads}): {new_group_size}"
                )
            return new_group_size
        else:
            return group_size

    def _get_classif_head_dim(self, input_dim, dim=None, ratio=None):
        """Helper function to easily get the layer dimension for the classification head

        Args:
            input_dim (_type_): _description_
            dim (_type_, optional): _description_. Defaults to None.
            ratio (_type_, optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_

        if dim is defined will directly use dim, even if you pass a ratio.
        If ratio is defined and not dim. Will use ratio to compute the correct
        dim from the head_input_dim
        """
        if dim is None and ratio is None:
            raise ValueError("Both `ratio` and `dim` cannot be None at the same time.")
        if ratio is not None and dim is None:
            dim = np.rint((np.cumprod(ratio) * input_dim)).astype(int).tolist()
        return dim
