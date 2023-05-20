import argparse
import inspect
import random
from ast import literal_eval
from dataclasses import dataclass

import numpy as np
import torch
import yaml

import wandb

TIMING_TABLE = {
    "msec": 1000,
    "sec": 1,
    "min": 1 / 60,
    "hour": 1 / 3600,
}


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def arguments_parsing(path):
    p = argparse.ArgumentParser()

    arguments = load_yaml(path)

    for name, argument in arguments.items():
        argument["type"] = (
            {**globals(), **__builtins__}.get(argument["type"], None)
            if "type" in argument
            else None
        )
        p.add_argument(f"--{name}", **argument)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save(path, model):
    torch.save(model.state_dict(), path)


def load(path, model):
    model.load_state_dict(torch.load(path, map_location="cpu"))


def wandb_logger(keys):
    def decorator(func):
        def wrap(args):
            config = {k: getattr(args, k) for k in keys if k in args.__dict__}
            config["source"] = args.dataset["domains"][args.source]
            config["target"] = args.dataset["domains"][args.target]
            wandb.init(
                project=f'{args.dataset["name"]}_{args.shot}',
                name=args.method,
                config=config,
            )
            func(args)
            wandb.finish()

        return wrap

    return decorator


class LR_Scheduler(object):
    def __init__(self, optimizer, num_iters, step=0, final_lr=None):
        # if final_lr: use cos scheduler, otherwise, use gamma scheduler
        self.final_lr = final_lr
        self.optimizer = optimizer
        self.iter = step
        self.num_iters = num_iters
        self.current_lr = optimizer.param_groups[-1]["lr"]

    def step(self):
        for param_group in self.optimizer.param_groups:
            base = param_group["base_lr"]
            self.current_lr = param_group["lr"] = (
                self.final_lr
                + 0.5
                * (base - self.final_lr)
                * (1 + np.cos(np.pi * self.iter / self.num_iters))
                if self.final_lr
                else base * ((1 + 0.0001 * self.iter) ** (-0.75))
            )
        self.iter += 1

    def refresh(self):
        self.iter = 0

    def get_lr(self):
        return self.current_lr


@dataclass
class BaseConfig:
    @classmethod
    def from_args(cls, args):
        return cls(
            **{
                k: v
                for k, v in args.__dict__.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class BaseTrainerConfig(BaseConfig):
    num_iters: int
    log_interval: int
    early: int
    eval_interval: int


@dataclass
class SLATrainerConfig(BaseTrainerConfig):
    warmup: int
    T: float
    alpha: float
    update_interval: int


@dataclass
class MetricMeter:
    counter: int = 0
    best_acc: float = 0
    best_val_acc: float = 0
    start_time: float = 0
