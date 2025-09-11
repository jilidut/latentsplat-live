# src/callbacks/gif_every_n_step.py
# import os, torch, math, imageio
from pathlib import Path
from pytorch_lightning.callbacks import Callback
from src.callbacks.generate_gif import GenerateGif

class GifEveryNStep(Callback):
    def __init__(self, every_n_steps=2500, **gif_kwargs):
        self.every_n_steps = every_n_steps
        self.gif_gen = GenerateGif(**gif_kwargs)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if (trainer.global_step + 1) % self.every_n_steps == 0:
            self.gif_gen.generate(trainer, pl_module)