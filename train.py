"""Train the ActionPlan model

Usage:
    python train.py                                # fresh run -> outputs/actionplan
    python train.py resume_dir=outputs/actionplan  # resume from last checkpoint

The full config lives in configs/train_actionplan.yaml and is frozen to
<run_dir>/config.json at startup, which generate.py and eval.py read later.
"""

import logging

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate
import pytorch_lightning as pl

import src.prepare  # noqa: F401 - sets torch/warning defaults
from src.config import read_config, save_config

logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="train_actionplan", version_base="1.3")
def train(cfg: DictConfig):
    # Resume from an existing run directory if requested
    ckpt = None
    if cfg.resume_dir is not None:
        resume_dir = cfg.resume_dir
        max_epochs = cfg.trainer.max_epochs
        assert cfg.ckpt is not None
        ckpt = cfg.ckpt
        cfg = read_config(resume_dir)
        cfg.trainer.max_epochs = max_epochs
        logger.info("Resuming training from: %s", resume_dir)
    else:
        config_path = save_config(cfg)
        logger.info("Training config saved to: %s", config_path)

    pl.seed_everything(cfg.seed)

    logger.info("Loading the dataloaders")
    val_split = "train" if cfg.data.get("val_same_as_train", False) else "val"
    train_dataset = instantiate(cfg.data, split="train")
    val_dataset = instantiate(cfg.data, split=val_split)

    train_dataloader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
    )
    val_dataloader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        collate_fn=val_dataset.collate_fn,
        shuffle=False,
    )

    logger.info("Loading the model")
    diffusion = instantiate(cfg.diffusion)

    logger.info("Training")
    trainer = instantiate(cfg.trainer)
    trainer.fit(diffusion, train_dataloader, val_dataloader, ckpt_path=ckpt)


if __name__ == "__main__":
    train()
