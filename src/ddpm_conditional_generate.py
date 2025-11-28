"""
This code train a conditional diffusion model on CIFAR.
It is based on @dome272.

@wandbcode{condition_diffusion}
"""

import argparse
import logging
import os
from types import SimpleNamespace

import torch

from diff_modules import DiffusionVAE
from diff_utils import set_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = SimpleNamespace(
    run_name="DDPM_conditional_VAE",
    seed=42,
    epochs=1,
    num_classes=27,
    noise_steps=1000,
    img_size=256,
    batch_size=4,
    dataset_path="Birdnet_conf_files_images_split",
    img_folder="diffusion_samples",
    device="cuda",
    slice_size=1,
    train_folder="train",
    val_folder="test",
    do_validation=True,
    fp16=True,
    log_every_epoch=10,
    num_workers=10,
    start_idx=0,
    lr=5e-3,
    load_model=False,
    num_samples=50,
    sav_denoise_path=None,
)


logging.basicConfig(
    format="%(asctime)s - %(levelname)s: %(message)s",
    level=logging.INFO,
    datefmt="%I:%M:%S",
)


def parse_args(config):
    parser = argparse.ArgumentParser(description="Process hyper-parameters")
    parser.add_argument("--seed", type=int, default=config.seed, help="random seed")
    parser.add_argument("--img_size", type=int, default=config.img_size, help="image size")
    parser.add_argument("--num_classes", type=int, default=config.num_classes, help="number of classes")
    parser.add_argument("--device", type=str, default=config.device, help="device")
    parser.add_argument("--slice_size", type=int, default=config.slice_size, help="slice size")
    parser.add_argument("--noise_steps", type=int, default=config.noise_steps, help="noise steps")
    parser.add_argument("--load_model", type=bool, default=True, help="load model")
    parser.add_argument("--img_folder", type=str, default=config.img_folder, help="image folder")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=config.num_samples,
        help="number of samples per class",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=config.run_name,
        help="run name (where to load model)",
    )
    parser.add_argument("--dataset_path", type=str, default=config.dataset_path, help="dataset path")
    parser.add_argument("--start_idx", type=int, default=config.start_idx, help="start index")
    parser.add_argument(
        "--sav_denoise_path",
        type=str,
        default=config.sav_denoise_path,
        help="save denoise path",
    )
    args = vars(parser.parse_args())

    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == "__main__":
    parse_args(config)

    # seed everything
    set_seed(config.seed)

    if not os.path.exists(config.img_folder):
        os.makedirs(config.img_folder)

    if config.sav_denoise_path is not None:
        if not os.path.exists(config.sav_denoise_path):
            os.makedirs(config.sav_denoise_path)

    class_names = sorted(os.listdir(os.path.join(config.dataset_path, config.train_folder)))

    diffuser = DiffusionVAE(
        config.noise_steps,
        img_size=config.img_size,
        num_classes=config.num_classes,
        sav_denoise_path=config.sav_denoise_path,
        class_names=class_names,
    )

    diffuser.prepare(config)
    diffuser.load_model(config)
    for samp_i in range(config.start_idx, config.start_idx + config.num_samples):
        diffuser.gen_images(config.img_folder, samp_i)

    print("done!")
