"""
This code train a conditional diffusion model on CIFAR.
It is based on @dome272.

@wandbcode{condition_diffusion}
"""

import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext

import torch
from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar

from diff_utils import *
from diff_modules import UNet_conditional, EMA, DiffusionVAE

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = SimpleNamespace(
    run_name = "DDPM_conditional_VAE",
    seed = 42,
    epochs = 1,
    num_classes = 27,
    noise_steps=1000,
    img_size = 256,
    batch_size = 4,
    dataset_path = 'Birdnet_conf_files_images_split',##get_cifar(img_size=64),
    img_folder = "diffusion_samples",
    device = "cuda",
    slice_size = 1,
    train_folder = "train",
    val_folder = "test",
    do_validation = True,
    fp16 = True,
    log_every_epoch = 10,
    num_workers=10,
    start_idx = 0,
    lr = 5e-3,
    load_model = False,
    num_samples = 50,
    sav_denoise_path = None)


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    parser.add_argument('--img_size', type=int, default=config.img_size, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    parser.add_argument('--load_model', type=bool, default=True, help='load model')
    parser.add_argument('--img_folder', type=str, default=config.img_folder, help='image folder')
    parser.add_argument('--num_samples', type=int, default=config.num_samples, help='number of samples per class')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='run name (where to load model)')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='dataset path')
    parser.add_argument('--start_idx', type=int, default=config.start_idx, help='start index')
    parser.add_argument('--sav_denoise_path', type=str, default=config.sav_denoise_path, help='save denoise path')
    args = vars(parser.parse_args())

    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)

if __name__ == '__main__':
    parse_args(config)

    ## seed everything
    set_seed(config.seed)

    if not os.path.exists(config.img_folder):
        os.makedirs(config.img_folder)

    if config.sav_denoise_path is not None:
      if not os.path.exists(config.sav_denoise_path):
        os.makedirs(config.sav_denoise_path)

    class_names = sorted(os.listdir(os.path.join(config.dataset_path, config.train_folder)))

    diffuser = DiffusionVAE(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes, sav_denoise_path = config.sav_denoise_path, class_names = class_names)
    #diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes)
    #print(class_names)
    #for every image in confiv.img_folder, rename file e.g. gen_imgs_0_0_0 to gen_imgs_{class_names[i]}_0_0
    #for i in range(config.num_classes):
    #    for j in range(250):#config.num_samples):
    #      if os.path.exists(f"{config.img_folder}/gen_imgs_{class_names[i]}_{i}_{j}.png"):
    #        os.rename(f"{config.img_folder}/gen_imgs_{class_names[i]}_{i}_{j}.png", f"{config.img_folder}/{class_names[i]}_gen_imgs_{i}_{j}.png")
          #if os.path.exists(f"{config.img_folder}/gen_imgs_class{i}_{i}_{j}.png"):
          #  os.rename(f"{config.img_folder}/gen_imgs_class{i}_{i}_{j}.png", f"{config.img_folder}/gen_imgs_{class_names[i]}_{i}_{j}.png")

    diffuser.prepare(config)
    diffuser.load_model(config)
    for samp_i in range(config.start_idx, config.start_idx + config.num_samples):
      diffuser.gen_images(config.img_folder, samp_i)

print("done!")
