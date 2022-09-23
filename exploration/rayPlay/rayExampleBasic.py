"""Simple example using RayAccelerator and Ray Tune"""
import os
import tempfile

from pl_bolts.datamodules.mnist_datamodule import MNISTDataModule

import pytorch_lightning as pl
import ray
from ray import tune
from ray_lightning.tune import TuneReportCallback, get_tune_resources
from ray_lightning import RayStrategy
import os
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import pytorch_lightning as pl
from pytorch_lightning.strategies import Strategy
from pytorch_lightning import LightningModule, Callback, Trainer, \
    LightningDataModule

import torchmetrics



class LightningMNISTClassifier(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super(LightningMNISTClassifier, self).__init__()

        self.data_dir = data_dir or os.getcwd()
        self.lr = config["lr"]
        layer_1, layer_2 = config["layer_1"], config["layer_2"]
        self.batch_size = config["batch_size"]

        # mnist images are (1, 28, 28) (channels, width, height)
        self.layer_1 = torch.nn.Linear(28 * 28, layer_1)
        self.layer_2 = torch.nn.Linear(layer_1, layer_2)
        self.layer_3 = torch.nn.Linear(layer_2, 10)
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
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
                use_gpu=False,
                callbacks=None):
    # Make sure data is downloaded on all nodes.
    def download_data():
        from filelock import FileLock
        with FileLock(os.path.join(data_dir, ".lock")):
            MNISTDataModule(data_dir=data_dir).prepare_data()

    model = LightningMNISTClassifier(config, data_dir)

    callbacks = callbacks or []

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        callbacks=callbacks,
        progress_bar_refresh_rate=0,
        strategy=RayStrategy(
            num_workers=num_workers, use_gpu=use_gpu, init_hook=download_data))
    dm = MNISTDataModule(
        data_dir=data_dir, num_workers=1, batch_size=config["batch_size"])
    trainer.fit(model, dm)


def tune_mnist(data_dir,
               num_samples=10,
               num_epochs=10,
               num_workers=1,
               use_gpu=False):
    config = {
        "layer_1": tune.choice([32, 64, 128]),
        "layer_2": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32, 64, 128]),
    }

    # Add Tune callback.
    metrics = {"loss": "ptl/val_loss", "acc": "ptl/val_accuracy"}
    callbacks = [TuneReportCallback(metrics, on="validation_end")]
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

data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")
tune_mnist(data_dir, 3, 4, 12, True)


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--num-workers",
#         type=int,
#         help="Number of training workers to use.",
#         default=1)
#     parser.add_argument(
#         "--use-gpu", action="store_true", help="Use GPU for training.")
#     parser.add_argument(
#         "--num-samples",
#         type=int,
#         default=10,
#         help="Number of samples to tune.")
#     parser.add_argument(
#         "--num-epochs",
#         type=int,
#         default=10,
#         help="Number of epochs to train for.")
#     parser.add_argument(
#         "--smoke-test", action="store_true", help="Finish quickly for testing")
#     parser.add_argument(
#         "--address",
#         required=False,
#         type=str,
#         help="the address to use for Ray")
#     args, _ = parser.parse_known_args()

#     num_epochs = 1 if args.smoke_test else args.num_epochs
#     num_workers = 1 if args.smoke_test else args.num_workers
#     use_gpu = False if args.smoke_test else args.use_gpu
#     num_samples = 1 if args.smoke_test else args.num_samples

#     if args.smoke_test:
#         ray.init(num_cpus=2)
#     else:
#         ray.init(address=args.address)

#     data_dir = os.path.join(tempfile.gettempdir(), "mnist_data_")
#     tune_mnist(data_dir, num_samples, num_epochs, num_workers, use_gpu)