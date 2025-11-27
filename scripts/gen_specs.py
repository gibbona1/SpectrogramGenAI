import numpy as np
from helpers import SpecDataset, get_neal_data
from torch.utils.data import DataLoader

lab_path = "datasets/neal_labels_remapped.csv"
wav_path = "datasets/neal_data"
test_path = "datasets/neal_images"
# lab_path = 'model_output_loc_merge.csv'
# wav_path = 'Birdnet_conf_files'
specdata = np.load("datasets/specdata.npz", allow_pickle=True)

# df_cols = ['index0'] + list(pd.read_csv(lab_path).columns)
# train_df = pd.DataFrame(specdata['train_df'], columns=df_cols)
# test_df = pd.DataFrame(specdata['test_df'], columns=df_cols)

pop_classes = specdata["categories"]
num_classes = len(pop_classes)

df = get_neal_data(lab_path, wav_path, pop_classes)

# pop_classes = specdata['categories']
# num_classes = len(pop_classes)

batch_size = 16

loader = DataLoader(
    dataset=SpecDataset(df, wav_path, pop_classes, sav_folder=test_path),
    batch_size=batch_size,
    shuffle=True,
)

for i, data in enumerate(loader):
    print(f"{i}/{len(loader)}")
    # if i > 2:
    #  break
