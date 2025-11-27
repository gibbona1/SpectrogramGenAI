import os

import numpy as np
import pandas as pd
import torch
from helpers import adjust_model, get_neal_data, load_ensemble, tic, toc, train_model

synth_data = [0, 50, 100, 150, 200, 250]
knowledge_dist = True
large_data = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
is_cuda = device.type == "cuda"

lab_path = "datasets/model_output_loc_merge.csv"
wav_path = "datasets/Birdnet_conf_files"
test_path = "datasets/neal_images"
# gen_path  = 'spec_gan2/generated_images' #change for noisered or aind
gen_path = "synthetic_data/diffusion_samples"

specdata = np.load("datasets/specdata.npz", allow_pickle=True)

pop_classes = specdata["categories"]
num_classes = len(pop_classes)

if large_data:
    im_path = "datasets/Birdnet_all_files_27class_images"
    largeinfo = np.load("datasets/birdnet_train_val_split.npz", allow_pickle=True)
    df_cols = largeinfo["columns"]
    train_df = pd.DataFrame(largeinfo["train_df"], columns=list(df_cols))  # .iloc[:16]
    val_df = pd.DataFrame(largeinfo["val_df"], columns=list(df_cols))  # .iloc[:16]
    train_df = train_df.loc[train_df["confidence"] >= 0.25]
    val_df = val_df.loc[val_df["confidence"] >= 0.25]
    train_df["formatted_filename"] = train_df.apply(
        lambda row: f'{row["file_name"]}_{int(row["begin_time"])}_{int(row["begin_time"])}.png',
        axis=1,
    )
    val_df["formatted_filename"] = val_df.apply(
        lambda row: f'{row["file_name"]}_{int(row["begin_time"])}_{int(row["begin_time"])}.png',
        axis=1,
    )
    train_df = train_df.loc[train_df["formatted_filename"].isin(os.listdir(im_path))]
    val_df = val_df.loc[val_df["formatted_filename"].isin(os.listdir(im_path))]
else:
    im_path = "datasets/Birdnet_conf_files_images"
    df_cols = pd.read_csv(lab_path).columns
    train_df = pd.DataFrame(specdata["train_df"], columns=["index0"] + list(df_cols))  # .iloc[:16]
    val_df = pd.DataFrame(specdata["test_df"], columns=["index0"] + list(df_cols))  # .iloc[:16]

test_df = get_neal_data("datasets/neal_labels_remapped.csv", test_path, pop_classes)  # .iloc[:16]


def redistribute_datasets(train_df, val_df, test_df, pop_classes):
    # Create copies to avoid modifying original dataframes
    train_df_new = train_df.copy()
    val_df_new = val_df.copy()
    test_df_new = test_df.copy()

    # For each class in pop_classes
    for class_name in pop_classes:
        # Count occurrences in test_df
        class_count = len(test_df_new[test_df_new["common_name"] == class_name])

        if class_count <= 3:
            # Do nothing if 0-3 samples
            continue

        elif 4 <= class_count <= 10:
            # Move 2 to train and 2 to val
            samples_to_move = test_df_new[test_df_new["common_name"] == class_name].sample(n=4)
            train_samples = samples_to_move.iloc[:2]
            val_samples = samples_to_move.iloc[2:]

        elif 11 <= class_count <= 50:
            # Move 5 to train and 1 to val
            samples_to_move = test_df_new[test_df_new["common_name"] == class_name].sample(n=6)
            train_samples = samples_to_move.iloc[:5]
            val_samples = samples_to_move.iloc[5:]

        elif 51 <= class_count <= 100:
            # Move 25 to train and 5 to val
            samples_to_move = test_df_new[test_df_new["common_name"] == class_name].sample(n=30)
            train_samples = samples_to_move.iloc[:25]
            val_samples = samples_to_move.iloc[25:]

        elif class_count > 100:
            # Move 50 to train and 10 to val
            samples_to_move = test_df_new[test_df_new["common_name"] == class_name].sample(n=60)
            train_samples = samples_to_move.iloc[:50]
            val_samples = samples_to_move.iloc[50:]

        # Append samples to train_df and val_df
        train_df_new = pd.concat([train_df_new, train_samples])
        val_df_new = pd.concat([val_df_new, val_samples])

        # Remove moved samples from test_df
        test_df_new = test_df_new.drop(samples_to_move.index)
    print("redistributed datasets")
    return train_df_new, val_df_new, test_df_new


batch_size = 16
output_dir = "results"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

model_name_dict = {"mobilenet": "mobilenet_v2", "vgg": "vgg16", "resnet": "resnet18"}

for model_name in ["resnet", "vgg", "mobilenet", "custom", "ensemble"]:
    tic()
    m_name = model_name_dict.get(model_name, model_name)
    print(f"running {m_name}")
    for synth in synth_data:
        print(f"synth: {synth}")
        if m_name == "ensemble":
            model = load_ensemble(synth, num_classes, device, knowledge_dist=knowledge_dist)
        else:
            model = adjust_model(m_name, num_classes)
        train_model(
            model,
            (model_name, synth, False, False, False, knowledge_dist),
            (train_df, val_df, test_df, gen_path),
            (im_path, test_path, pop_classes, device, output_dir),
            num_epochs=25,
            save_interval=1,
            batch_size=batch_size,
        )
    toc()
print("Done")
