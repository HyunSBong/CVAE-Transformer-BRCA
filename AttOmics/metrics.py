import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import torch
from pycox.evaluation.concordance import concordance_td
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))


def _ci_update_deephit(survival, time, event, time_idx):

    if survival.size(1) != time.size(0) and time.size(0) != event.size(0):
        raise ValueError(
            f"Expected the same number of elements in `time` and `event` tensor but received {time.numel()} and {event.numel()}"
        )

    if survival.ndim != 2:
        raise ValueError("Expected survival to be a 2d tensor.")

    if time.ndim > 1 or event.ndim > 1:
        raise ValueError(
            f"Expected both `risk`, `time` and  `event` tensor to be 1d, but got tensors with dimension {time.ndim} and {event.ndim}"
        )

    return survival, time, event, time_idx


def _ci_update(
    risk: Tensor, time: Tensor, event: Tensor
) -> Tuple[Tensor, Tensor, Tensor]:
    if risk.ndim > 1:
        risk = risk.squeeze()

    if risk.ndim > 1 or time.ndim > 1 or event.ndim > 1:
        raise ValueError(
            f"Expected both `risk`, `time` and  `event` tensor to be 1d, but got tensors with dimension {risk.ndim}, {time.ndim} and {event.ndim}"
        )
    if (
        risk.numel() != time.numel()
        or risk.numel() != event.numel()
        or time.numel() != event.numel()
    ):
        raise ValueError(
            f"Expected the same number of elements in `time` and `event` tensor but received {time.numel()} and {event.numel()}"
        )
    return risk, time, event


def _ci_compute(
    risk: Tensor,
    time: Tensor,
    event: Tensor,
    w: Tensor,
    reorder: bool = False,
    ties: bool = True,
) -> Tensor:
    with torch.no_grad():
        if reorder:
            time, idx = time.sort(dim=0, descending=True)
            event = event[idx]
            risk = risk[idx]
            w = w[idx]

        Tx, Ty = torch.meshgrid(time, time, indexing="xy")
        riskx, risky = torch.meshgrid(risk, risk, indexing="xy")
        Ex, Ey = torch.meshgrid(event, event, indexing="xy")

        Tcomp = torch.gt(Ty, Tx)
        Tcomp2 = torch.eq(Tx, Ty).mul(torch.ones_like(Ey).sub(Ey)).mul(Ex)
        risk_comp = torch.gt(riskx, risky)
        concordant = (
            (
                Tcomp.logical_and(risk_comp).mul(Ex).fill_diagonal_(0)
                + Tcomp2.logical_and(risk_comp)
            )
            .sum(dim=0)
            .mul(w)
            .sum()
        )
        pairs = Tcomp.mul(Ex).fill_diagonal_(0).add(Tcomp2).sum(dim=0).mul(w).sum()
        if ties:
            risk_eq = torch.eq(riskx, risky)
            tied = (
                (
                    Tcomp.logical_and(risk_eq).mul(Ex).fill_diagonal_(0)
                    + Tcomp2.logical_and(risk_eq)
                )
                .sum(dim=0)
                .mul(w)
                .sum()
            )
        else:
            tied = torch.tensor(0, dtype=time.dtype, device=time.device)

        return concordant.add(tied.mul(0.5)).div(pairs)


def concordance_index(
    risk: Tensor, time: Tensor, event: Tensor, w: Tensor = None, reorder: bool = False
) -> Tensor:
    risk, time, event = _ci_update(risk, time, event)
    if w is None:
        w = torch.ones_like(risk)
    return _ci_compute(risk, time, event, w=w, reorder=reorder)


class TimeDependentConcordanceIndex(Metric):
    def __init__(
        self,
        reorder: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.reorder = reorder

        self.add_state("survival", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=[], dist_reduce_fx="cat")
        self.add_state("event", default=[], dist_reduce_fx="cat")
        self.add_state("time_idx", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `ConcordanceIndex` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(
        self, survival: Tensor, time: Tensor, event: Tensor, time_idx: Tensor
    ) -> None:
        survival, time, event, time_idx = _ci_update_deephit(
            survival, time, event, time_idx
        )

        self.survival.append(survival)
        self.time.append(time)
        self.event.append(event)
        self.time_idx.append(time_idx)

    def compute(self) -> Tensor:
        survival = dim_zero_cat(self.survival)
        time = dim_zero_cat(self.time)
        event = dim_zero_cat(self.event)
        survival_idx = dim_zero_cat(self.time_idx)

        return concordance_td(
            time.detach().cpu().numpy(),
            event.detach().cpu().numpy(),
            survival.detach().cpu().numpy(),
            survival_idx.detach().cpu().numpy(),
            method="adj_antolini",
        )


class ConcordanceIndex(Metric):
    """
    \frac{\text{concordant pairs}}{\text{disconcordant pairs} + \text{concordant pairs}} = \frac{\sum_{ij} \mathbf{1}_{T_j < T_i}\cdot \mathbf{1}_{\eta_j>\eta_i}\cdot \delta_j}{\sum_ij \mathbf{1}_{T_j < T_i}\cdot\delta_j}
    """

    def __init__(
        self,
        reorder: bool = False,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        self.reorder = reorder

        self.add_state("risk", default=[], dist_reduce_fx="cat")
        self.add_state("time", default=[], dist_reduce_fx="cat")
        self.add_state("event", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `ConcordanceIndex` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(self, risk: Tensor, time: Tensor, event: Tensor) -> None:
        risk, time, event = _ci_update(risk, time, event)

        self.risk.append(risk)
        self.time.append(time)
        self.event.append(event)

    def compute(self) -> Tensor:
        risk = dim_zero_cat(self.risk)
        time = dim_zero_cat(self.time)
        event = dim_zero_cat(self.event)
        w = torch.ones_like(time)
        return _ci_compute(risk, time, event, w=w, reorder=self.reorder)
