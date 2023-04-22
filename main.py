import os
import sys
from ast import literal_eval

import configargparse

sys.path.append("src")

from dataset import DataIterativeLoader
from trainer import BaseDATrainer, UnlabeledDATrainer, get_trainer
from util import set_seed, wandb_logger

# from trainer import HyperParameters, ConfigParameters


def arguments_parsing():
    p = configargparse.ArgumentParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )
    p.add("--config", is_config_file=True, default="./config.yaml")
    p.add("--device", type=str, default="5")
    p.add(
        "--method",
        type=str,
        default="mme_SLA",
        choices=["base", "base_SLA", "mme", "mme_SLA", "cdac", "cdac_SLA"],
    )

    p.add("--dataset", type=str, default="OfficeHome")
    p.add("--source", type=int, default=0)
    p.add("--target", type=int, default=1)

    # training settings
    p.add("--seed", type=int, default=19980802)
    p.add("--bsize", type=int, default=24)
    p.add("--num_iters", type=int, default=8000)
    p.add("--shot", type=str, default="3shot", choices=["1shot", "3shot"])
    p.add("--alpha", type=float, default=0.3)

    p.add("--eval_interval", type=int, default=500)
    p.add("--log_interval", type=int, default=100)
    p.add("--update_interval", type=int, default=500)
    p.add("--early", type=int, default=0)
    p.add("--warmup", type=int, default=500)
    # configurations
    p.add("--dataset_cfg", type=literal_eval)

    # optimizer
    p.add("--lr", type=float, default=0.01)
    p.add("--momentum", type=float, default=0.9)
    p.add("--weight_decay", type=float, default=5e-4)
    p.add("--T", type=float, default=0.6)
    return p.parse_args()


@wandb_logger(
    keys=[
        "source",
        "target",
        "seed",
        "num_iters",
        "alpha",
        "T",
        "update_interval",
        "lr",
        "warmup",
    ]
)
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    set_seed(args.seed)

    match args.method.split("_"):
        case "cdac", *_:
            loaders = DataIterativeLoader(args, strong_transform=True)
        case _:
            loaders = DataIterativeLoader(args, strong_transform=False)

    match args.method.split("_"):
        case "base", *label_trick:
            trainer = get_trainer(BaseDATrainer, label_trick)(loaders, args)
        case ("mme" | "cdac") as unlabeled_method, *label_trick:
            trainer = get_trainer(UnlabeledDATrainer, label_trick)(
                loaders, args, unlabeled_method=unlabeled_method
            )

    trainer.train()


if __name__ == "__main__":
    args = arguments_parsing()
    # replace the configuration
    args.dataset = args.dataset_cfg[args.dataset]

    main(args)
