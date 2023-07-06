import math
import sys
import tempfile
from pathlib import Path

path_root = Path(__file__)
sys.path.append(str(path_root))

import logging

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sn
import torch
import torch.nn.functional as F
import torchmetrics
from pytorch_lightning.loggers.base import DummyLogger
from pytorch_lightning.utilities.cli import instantiate_class
from torch import nn

from losses import PartialLogLikelihood
from metrics import ConcordanceIndex, TimeDependentConcordanceIndex

logger = logging.getLogger(__name__)

mpl.rcParams["savefig.bbox"] = "tight"

def get_number_parameters(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


class BaseModel(pl.LightningModule):
    def __init__(
        self,
        optimizer_init: dict,
        scheduler_init: dict,
        label_type: str,
        num_classes: int,
        class_weights=None,
        train_data=None,
    ):
        super().__init__()

        self.surv_label = "survival"
        self.optimizer_init = optimizer_init
        self.scheduler_init = scheduler_init
        self.label_type = label_type
        self.class_weights = class_weights
        self.num_classes = num_classes
        self.train_data = train_data

        self.train_metrics = self.metrics().clone(prefix="train_")
        self.val_metrics = self.metrics().clone(prefix="val_")
        if label_type not in self.surv_label:
            self.val_cm = torchmetrics.ConfusionMatrix(self.num_classes)

        self.init_model()
        self.loss_fn = self.configure_loss()

    def configure_loss(self):
        if self.label_type == "survival":
            return PartialLogLikelihood
        else:
            return nn.NLLLoss(weight=torch.Tensor(self.class_weights))

    def metrics(self):
        if self.label_type == "survival":
            return torchmetrics.MetricCollection([ConcordanceIndex(reorder=True)])

        elif self.label_type == "deephit":
            return torchmetrics.MetricCollection(
                [TimeDependentConcordanceIndex(reorder=True)]
            )
        else:
            return torchmetrics.MetricCollection(
                [
                    torchmetrics.Accuracy(num_classes=self.num_classes),
                    torchmetrics.F1(num_classes=self.num_classes),
                    torchmetrics.AUROC(num_classes=self.num_classes),
                ]
            )

    def init_model(self):
        raise NotImplementedError

    def on_train_start(self):
        if self.trainer.datamodule is not None:
            n_train_examples = self.trainer.datamodule.num_examples
        else:
            n_train_examples = len(self.trainer.train_dataloader.loaders.dataset)
            if self.trainer.train_dataloader.loaders.drop_last:
                n_train_examples -= (
                    n_train_examples % self.trainer.train_dataloader.loaders.batch_size
                )
        self.logger.log_hyperparams(
            {
                "optimizer": self.optimizer_init["class_path"],
                "scheduler": self.scheduler_init["class_path"],
                "lr": self.optimizer_init["init_args"]["lr"],
                "model": self.__class__.__name__,
                "n_parameters": get_number_parameters(self),
                "n_train_examples": n_train_examples,
            }
        )

    def validation_epoch_end(self, val_step_outs):
        if self.hparams.label_type not in self.surv_label:
            preds = torch.cat([x["preds"] for x in val_step_outs], dim=0)
            target = torch.cat([x["target"] for x in val_step_outs], dim=0)
            cm = self.val_cm(preds, target)
            if self.trainer.datamodule is not None:
                label = self.trainer.datamodule.label_str
            else:
                label = self.label_str
            g = sn.heatmap(
                cm.cpu(),
                annot=True,
                fmt="g",
                cbar=False,
                square=True,
                xticklabels=label,
                yticklabels=label,
                cmap="Blues",
            )
            if not isinstance(self.logger, DummyLogger) or len(self.logger) != 0:
                self.logger.experiment.log_figure(
                    self.logger.run_id,
                    g.figure,
                    f"val_confusion_matrix-{self.current_epoch}.png",
                )
            self.val_cm.reset()

            plt.close()

    def test_epoch_end(self, outputs):
        # if there is only one dataloader, one level is remove recreate it
        if len(self.trainer.test_dataloaders) == 1:
            outputs = [outputs]
        logger.debug(self.trainer.test_dataloaders)
        logger.debug(f"Number of test dataloaders: {len(outputs)}")
        for i, test_step_outs in enumerate(outputs):
            logger.debug(f"Number of steps for dataloader {i}: {len(test_step_outs)}")
            prefix = self.trainer.test_dataloaders[i].dataset.name
            preds = torch.cat([x["preds"] for x in test_step_outs], dim=0)
            target = torch.cat([x["target"] for x in test_step_outs], dim=0)
            logger.debug(f"Predictions shape : {preds.shape}")
            logger.debug(
                f"Dataset dim: {len(self.trainer.test_dataloaders[i].dataset)}"
            )
            # save to a temp location the preds
            with tempfile.TemporaryDirectory() as tmp_dir:
                if not isinstance(self.logger, DummyLogger) or len(self.logger) != 0:
                    local_path = Path(tmp_dir, f"{prefix}_predictions.pt")
                    local_path2 = Path(tmp_dir, f"{prefix}_targets.pt")
                    local_path3 = Path(tmp_dir, f"{prefix}_events.pt")
                    local_path4 = Path(tmp_dir, f"{prefix}_attention.pt")
                    local_path6 = Path(tmp_dir, f"{prefix}_sampleID.npy")
                    torch.save(preds, local_path)
                    torch.save(target, local_path2)
                    logger.debug(
                        f"Uploading model prediction on the test dataloader {i}"
                    )
                    self.logger.experiment.log_artifact(self.logger.run_id, local_path)
                    logger.debug("Uploading associated label")
                    self.logger.experiment.log_artifact(self.logger.run_id, local_path2)
                    sampleID = self.trainer.datamodule.test_dataset.sampleID
                    np.save(local_path6, sampleID)
                    self.logger.experiment.log_artifact(self.logger.run_id, local_path6)
                    if test_step_outs[0]["event"] is not None:
                        event = torch.cat([x["event"] for x in test_step_outs], dim=0)
                        torch.save(event, local_path3)
                        logger.info("Uploading associated events")
                        self.logger.experiment.log_artifact(
                            self.logger.run_id, local_path3
                        )

                    if "attention" in test_step_outs[0].keys():
                        attention_map = torch.cat(
                            [x["attention"] for x in test_step_outs], dim=0
                        )
                        torch.save(attention_map, local_path4)
                        logger.debug(
                            f"Uploading attention map for the test dataloader {i}"
                        )
                        self.logger.experiment.log_artifact(
                            self.logger.run_id, local_path4
                        )

            if self.hparams.label_type not in self.surv_label:
                cm = self.val_cm(preds, target)

                label = self.trainer.datamodule.label_str
                g = sn.heatmap(
                    cm.cpu(),
                    annot=True,
                    fmt="g",
                    cbar=False,
                    square=True,
                    xticklabels=label,
                    yticklabels=label,
                    cmap="Blues",
                )
                if not isinstance(self.logger, DummyLogger) or len(self.logger) != 0:
                    logger.debug(
                        f"Uploading ConfusionMatrix for the test dataloader {i}"
                    )
                    self.logger.experiment.log_figure(
                        self.logger.run_id, g.figure, f"{prefix}_confusion_matrix.png"
                    )
                    self.logger.experiment.log_text(
                        self.logger.run_id, ",".join(label), "class_names.txt"
                    )
                self.val_cm.reset()

                plt.close()
        group_name = getattr(self, "group_name", None)
        if group_name is not None and (
            not isinstance(self.logger, DummyLogger) or len(self.logger) != 0
        ):
            self.logger.experiment.log_text(
                self.logger.run_id, ",".join(group_name), "groups_name.txt"
            )

    def configure_optimizers(self):
        opt = instantiate_class(self.parameters(), self.optimizer_init)
        scheduler = {
            "scheduler": instantiate_class(opt, self.scheduler_init),
            "monitor": "val_loss",
            "frequency": 1,
            "interval": "epoch",
        }
        return {"optimizer": opt, "lr_scheduler": scheduler}


class BaseModelSingleOmics(BaseModel):
    def __init__(
        self,
        optimizer_init: dict,
        scheduler_init: dict,
        label_type: str,
        num_classes: int,
        class_weights=None,
        train_data=None,
    ):

        super().__init__(
            optimizer_init=optimizer_init,
            scheduler_init=scheduler_init,
            label_type=label_type,
            num_classes=num_classes,
            class_weights=class_weights,
            train_data=train_data,
        )

    def forward(self, x, y, event=None):
        y_hat = self.model(x)
        if self.hparams.label_type == "survival":
            loss = self.loss_fn(y_hat, y, event)
        else:
            y_hat = F.log_softmax(y_hat, dim=1)
            loss = self.loss_fn(y_hat, y)
        return y_hat, loss

    def compute_metrics(self, stage, y_hat, y, event=None):
        metrics_fn = self.train_metrics if stage == "train" else self.val_metrics
        if self.hparams.label_type == "survival":
            out_metrics = metrics_fn(y_hat, y, event)
        else:
            out_metrics = metrics_fn(y_hat.exp(), y)  # classification use proba
        return out_metrics

    def predict_step(self, batch, batch_idx):
        x = batch.get("x", None)
        y = batch.get("label", None)

        event = batch.get("event", None)

        y_hat, _ = self.forward(
            x,
            y,
            event=event,
        )

        if self.hparams.label_type == "survival":
            return y_hat
        else:
            return y_hat.exp()

    def training_step(self, batch, batch_idx):
        x = batch.get("x", None)
        y = batch.get("label", None)

        event = batch.get("event", None)

        y_hat, loss = self.forward(
            x,
            y,
            event=event,
        )
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        out_metrics = self.compute_metrics("train", y_hat, y, event)
        self.log_dict(
            out_metrics,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        return {"loss": loss, "preds": y_hat.detach()}

    def validation_step(self, batch, batch_index):
        x = batch.get("x", None)
        y = batch.get("label")

        event = batch.get("event", None)

        y_hat, loss = self.forward(
            x,
            y,
            event=event,
        )
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        out_metrics = self.compute_metrics("val", y_hat, y, event)
        self.log_dict(
            out_metrics,
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        return {
            "val_loss": loss,
            "preds": y_hat.detach(),
            "target": y,
            "event": event,
        }

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        x = batch.get("x", None)
        y = batch.get("label")

        event = batch.get("event", None)
        y_hat, loss = self.forward(
            x,
            y,
            event=event,
        )

        return {
            "loss": loss,
            "preds": y_hat.detach(),
            "target": y,
            "event": event,
        }
