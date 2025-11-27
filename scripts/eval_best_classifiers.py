import os

import numpy as np
import pandas as pd
import torch
from helpers import (
    adjust_model,
    eval_best_model,
    get_neal_data,
    load_ensemble,
    tic,
    toc,
)

synth_data = [0, 50, 100, 150, 200, 250]
knowledge_dist = True
large_data = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_cuda = device.type == "cuda"
data_folder = "datasets"
lab_path = os.path.join(data_folder, "model_output_loc_merge.csv")
wav_path = os.path.join(data_folder, "Birdnet_conf_files")
im_path = os.path.join(data_folder, "Birdnet_conf_files_images")
test_path = os.path.join(data_folder, "neal_images")
# gen_path  = 'spec_gan2/generated_images' #change for noisered or aind
gen_path = "synthetic_data/diffusion_samples"

specdata = np.load(os.path.join(data_folder, "specdata.npz"), allow_pickle=True)

pop_classes = specdata["categories"]
num_classes = len(pop_classes)

if large_data:
    im_path = "datasets/Birdnet_all_files_27class_images"
    largeinfo = np.load("datasets/birdnet_train_val_split.npz", allow_pickle=True)
    df_cols = largeinfo["columns"]
    val_df = pd.DataFrame(largeinfo["val_df"], columns=list(df_cols))
    val_df["formatted_filename"] = val_df.apply(
        lambda row: f'{row["file_name"]}_{int(row["begin_time"])}_{int(row["begin_time"])}.png',
        axis=1,
    )
    val_df = val_df.loc[val_df["formatted_filename"].isin(os.listdir(im_path))]
else:
    im_path = "datasets/Birdnet_conf_files_images"
    df_cols = pd.read_csv(lab_path).columns
    val_df = pd.DataFrame(specdata["test_df"], columns=["index0"] + list(df_cols))
test_df = get_neal_data(os.path.join(data_folder, "neal_labels_remapped.csv"), test_path, pop_classes)

print(f"val_df: {val_df.shape}; test_df: {test_df.shape}")

batch_size = 16
output_dir = "eval_results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_name_dict = {"mobilenet": "mobilenet_v2", "vgg": "vgg16", "resnet": "resnet18"}

for model_name in ["mobilenet", "vgg", "resnet", "custom", "ensemble"]:
    tic()
    m_name = model_name_dict.get(model_name, model_name)
    print(f"running {m_name}")
    for synth in synth_data:
        print(f"synth: {synth}")
        if m_name == "ensemble":
            model = load_ensemble(synth, num_classes, device, best=True, knowledge_dist=knowledge_dist)
        else:
            model = adjust_model(
                model_name,
                num_classes,
                best=True,
                synth=synth,
                knowledge_dist=knowledge_dist,
            )
        eval_best_model(
            model,
            (model_name, synth, False, False, False),
            (val_df, test_df),
            (im_path, test_path, pop_classes, device, output_dir),
            batch_size=batch_size,
            knowledge_dist=knowledge_dist,
        )
    toc()
print("Done")
