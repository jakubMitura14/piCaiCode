"""Simple example using RayAccelerator and Ray Tune"""
import os
import tempfile

import math
import torch

from datetime import datetime
import os
import tempfile
from glob import glob
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
from ray import air, tune
from ray.air import session
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback
from ray_lightning import RayShardedStrategy

ray.init(num_cpus=24)
data_dir = '/home/sliceruser/mnist'
MNISTDataModule(data_dir=data_dir).prepare_data()
num_cpus_per_worker=6
test_l_dir = '/home/sliceruser/test_l_dir'

class netaA(nn.Module):
    def __init__(self,
        config
    ) -> None:
        super().__init__()
        layer_1, layer_2 = config["layer_1"], config["layer_2"]
        self.model = nn.Sequential(
        torch.nn.Linear(28 * 28, layer_1),
        torch.nn.Linear(layer_1, layer_2),    
        torch.nn.Linear(layer_2, 10)
        )
    def forward(self, x):
        return self.model(x)



class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()

        self.data_dir = data_dir or os.getcwd()
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]

        self.accuracy = torchmetrics.Accuracy()
        self.netA= netaA(config)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x= self.netA(x)

        x = F.log_softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        acc = self.accuracy(logits, y)
        self.log("ptl/train_loss", loss)
        self.log("ptl/train_accuracy", acc)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y.long())
        acc = self.accuracy(logits, y)
        return {"val_loss": loss, "val_accuracy": acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("ptl/val_loss", avg_loss)
        self.log("ptl/val_accuracy", avg_acc)



def train_mnist(config,
                data_dir=None,
                num_epochs=10,
                num_workers=1,
                use_gpu=True,
                callbacks=None):

    model = LightningMNISTClassifier(config, data_dir)

    callbacks = callbacks or []
    print(" aaaaaaaaaa  ")
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        progress_bar_refresh_rate=0,
        strategy=RayStrategy(
            num_workers=num_workers, use_gpu=use_gpu))#, init_hook=download_data
    dm = MNISTDataModule(
        data_dir=data_dir, num_workers=2, batch_size=config["batch_size"])
    trainer.fit(model, dm)


def tune_mnist(data_dir,
               num_samples=2,
               num_epochs=10,
               num_workers=2,
               use_gpu=True):
    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    # Add Tune callback.
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCheckpointCallback(metrics, on="validation_end",filename="checkpointtt")]
    trainable = tune.with_parameters(
        train_mnist,
        data_dir=data_dir,
        num_epochs=num_epochs,
        num_workers=num_workers,
        use_gpu=use_gpu,
        callbacks=callbacks)
    analysis = tune.run(
        trainable,
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        resources_per_trial=get_tune_resources(
            num_workers=num_workers, use_gpu=use_gpu),
        name="tune_mnist")

    print("Best hyperparameters found were: ", analysis.best_config)

tune_mnist(data_dir)
