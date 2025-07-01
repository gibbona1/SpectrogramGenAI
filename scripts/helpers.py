import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights, mobilenet_v2, MobileNet_V2_Weights, vgg16, VGG16_Weights
import librosa
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
from Classifiers import CustomModel, EnsembleModel
import time
from gan_models import ImageInpaintingModel
import torchmetrics
from sklearn.metrics import classification_report, confusion_matrix

model_name_dict = {
    'mobilenet': 'mobilenet_v2',
    'vgg': 'vgg16',
    'resnet': 'resnet18'
}

def tic():
    global start_time 
    start_time = time.time()

def toc():
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Elapsed Time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=0)
  
def softmax_np(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class SpecDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, root_dir, class_names, sav_folder = None):
        """
        Arguments:
            df (pd.DataFrame): Data with annotations.
            root_dir (string): Directory with all the images.
        """
        self.df       = df
        self.root_dir = root_dir
        self.class_names = class_names
        self.sav_folder = sav_folder
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx]
        wav_name = os.path.join(self.root_dir, row['file_name'])
        #print(row.file_name)
        if os.path.exists(row.file_name):
            spec = np.array(Image.open(row.file_name).convert('L'))
            spec = np.expand_dims(spec, 0)
            lab = list(self.class_names).index(row.common_name)
            
            return spec, lab
        elif os.path.exists(wav_name):
            wav, sr = librosa.core.load(wav_name, sr=None)
            wav_len = 6
            wav_sub = wav[int(int(row['begin_time'])*sr):int((int(row['begin_time'])+wav_len)*sr)]
            if(len(wav_sub)/sr != wav_len):
                wav_sub = wav[int((int(row['end_time'])-wav_len)*sr):int(int(row['end_time'])*sr)]
            nfft = 512
            
            spec = librosa.feature.melspectrogram(y=wav_sub, sr=sr, n_mels=256, hop_length=int(0.75*nfft))

            spec = librosa.power_to_db(spec, ref=np.max)[:,:256]
            
            lab  = list(self.class_names).index(row.common_name)
            if self.sav_folder:
                folder = self.sav_folder
                if not os.path.exists(folder):
                    os.makedirs(folder)
                im_name = os.path.join(folder, f'{row["file_name"]}_{int(row["begin_time"])}_{int(row["begin_time"])}.png')
                plt.imsave(im_name, spec) 
                return [], []
            spec = np.expand_dims(spec, 0) #channel first

            return spec, lab
        else:
            return None, None
            #return None, None

class SpecDatasetIm(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, root_dir, class_names):
        """
        Arguments:
            df (pd.DataFrame): Data with annotations.
            root_dir (string): Directory with all the images.
        """
        self.df       = df
        self.root_dir = root_dir
        self.class_names = class_names
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx]
        #im_name = os.path.join(self.root_dir, f'{row["file_name"][:-4]}_{int(row["begin_time"])}_{int(row["end_time"])}_{row.common_name}.png'))
        im_name = os.path.join(self.root_dir, f'{row["file_name"]}_{int(row["begin_time"])}_{int(row["begin_time"])}.png')
        #im_name = os.path.join(self.root_dir, f'{row["file_name"]}_{int(row["begin_time"])}.png')
        
        if os.path.exists(row.file_name):
            spec = np.array(Image.open(row.file_name).convert('L'))
            spec = np.expand_dims(spec, 0)
            lab = list(self.class_names).index(row.common_name)
            if 'embeddings' in row:
              #lab is a vector of numeric values
              #don't softmax
              emb = np.array([float(v) for v in row.embeddings.split(",")])
              lab = (lab, emb)
            return spec, lab
        elif os.path.exists(im_name):
            spec = np.array(Image.open(im_name).convert('L'))
            spec = np.expand_dims(spec, 0)
            lab = list(self.class_names).index(row.common_name)
            if 'embeddings' in row:
              #lab is a vector of numeric values
              #don't softmax
              emb = np.array([float(v) for v in row.embeddings.split(",")])
              lab = (lab, emb)
            
            return spec, lab
        else:
            return None, None

class SpecDatasetImInd(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, df, root_dir, class_names, indices):
        """
        Arguments:
            df (pd.DataFrame): Data with annotations.
            root_dir (string): Directory with all the images.
        """
        self.df       = df
        self.root_dir = root_dir
        self.class_names = class_names
        self.indices  = indices
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        row = self.df.iloc[idx]
        #im_name = os.path.join(self.root_dir, f'{row["file_name"][:-4]}_{int(row["begin_time"])}_{int(row["end_time"])}_{row.common_name}.png'))
        im_name = os.path.join(self.root_dir, f'{row["file_name"]}_{int(row["begin_time"])}_{int(row["begin_time"])}.png')
        lab = list(self.class_names).index(row.common_name)
        #print(pd.to_numeric(row[self.indices]).dtype)
        #import code; code.interact(local=dict(globals(), **locals()))
        a_inds = torch.tensor(pd.to_numeric(row[self.indices]).to_numpy())
        
        if os.path.exists(row.file_name):
            spec = np.array(Image.open(row.file_name).convert('L'))
            spec = np.expand_dims(spec, 0)
            return spec, lab, a_inds
        elif os.path.exists(im_name):
            spec = np.array(Image.open(im_name).convert('L'))
            spec = np.expand_dims(spec, 0)
            return spec, lab, a_inds
        else:
            return None, None, None

class MixDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, noise_df, bird_df, class_names, sr = None):
        """
        Arguments:
            df (pd.DataFrame): Data with annotations.
            root_dir (string): Directory with all the images.
        """
        self.noise_df = noise_df
        self.bird_df = bird_df
        self.class_names = class_names
        self.sr = sr
    
    def __len__(self):
        return min(len(self.noise_df), len(self.bird_df))
    
    def gen_wav(self, wav_name, wav_start = 0):
        sr = self.sr
        wav_len = 2**17/sr
        wav, _ = librosa.core.load(wav_name, sr=sr)
        #print(len(wav))
        wav_sub = wav[int(wav_start*sr):int((wav_start+wav_len)*sr)]
        if(len(wav_sub)/sr != wav_len):
            #print('here')
            #print(len(wav_sub)/sr, wav_len, wav_start)
            #start at the end of the file and get a slice of length wav_len*sr
            #wav_sub = wav[(int(len(wav)/sr-wav_len)*sr):int(int(len(wav)/sr)*sr)]
            wav_sub = wav[int(len(wav) - wav_len*sr):]
        #if still too short
        if(len(wav_sub)/sr < wav_len):
            wav_sub = np.repeat(wav_sub, wav_len//(len(wav_sub)/sr) + 1)[:int(wav_len*sr)]
        #print(1, wav_sub.min(), wav_sub.max(), sr)
        wav_sub = librosa.util.normalize(wav_sub)
        #print(2, wav_sub.min(), wav_sub.max(), sr)
        return wav_sub
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        noise_r, bird_r = self.noise_df.iloc[idx], self.bird_df.iloc[idx]
        #print(noise_r, bird_r)
        noise_wav, bird_wav = self.gen_wav(noise_r.path, bird_r.st), self.gen_wav(bird_r.path, bird_r.st)
        #print(len(noise_wav), len(bird_wav))
        def rescale_wav(x):
            x = (x-np.min(x))/(np.max(x)-np.min(x))
            return 2 * (x - 0.5)
        def samp_eps(a, b):
            return (b - a) * np.random.sample() + a
        eps = samp_eps(0.0, 0.3)
        mix_wav = eps * noise_wav + (1-eps) * bird_wav
        #normalize here instead, then put to -1, 1
        
        #noise_spec, bird_spec, mix_spec = self.gen_spec(noise_wav), self.gen_spec(bird_wav), self.gen_spec(mix_wav)
        #noise_wav, bird_wav, mix_wav = np.expand_dims(noise_wav, 0), np.expand_dims(bird_wav, 0), np.expand_dims(mix_wav, 0) #channel first
        noise_wav, bird_wav, mix_wav = rescale_wav(noise_wav), rescale_wav(bird_wav), rescale_wav(mix_wav)
        
        def gen_spec(wav_sub):
            sr = self.sr
            nfft = 512
            spec = librosa.feature.melspectrogram(y=wav_sub, sr=sr, n_mels=256, hop_length=int(0.75*nfft))
            spec = librosa.power_to_db(spec, ref=np.max)[:,:256]
            spec = np.expand_dims(spec, 0) #channel first
            
            m = (np.min(spec) + np.max(spec))/2
            spec = (spec - m)/m
            if m < 0:
                spec = -1*spec
            return spec
        
        noise_spec, bird_spec, mix_spec = gen_spec(noise_wav), gen_spec(bird_wav), gen_spec(mix_wav)
        
        lab  = list(self.class_names).index(bird_r.common_name)

        return noise_spec, bird_spec, mix_spec, lab

def prop_counts(df):
    return df['common_name'].value_counts(normalize=True).sort_index()

def resize_m1_1(x):
    m = (torch.min(x) + torch.max(x))/2
    x = (x - m)/m
    if m < 0:
        x = -1*x
    return x

def fast_resize_m1_1(x):
    min_values = x.reshape(x.shape[0],-1).min(dim=-1,keepdim=True)[0].unsqueeze(2).unsqueeze(3)
    max_values = x.reshape(x.shape[0],-1).max(dim=-1,keepdim=True)[0].unsqueeze(2).unsqueeze(3)
    m = max_values - min_values
    x = (x - min_values)/m
    x = (1 * (m >= 0) - 1 * (m < 0)) * 2 * (x - 0.5)
    return x

def adjust_model(model_name, num_classes, best = False, synth = 0, knowledge_dist=False):
    m_name = model_name_dict.get(model_name, model_name)

    if m_name == 'resnet18':
        model = resnet18(weights = ResNet18_Weights.IMAGENET1K_V1)
        for name, param in model.named_parameters():
            if name.startswith('layer4') or name.startswith('fc'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        model.fc = nn.Linear(512, num_classes) # assuming that the fc layer has 512 neurons, otherwise change it 
    elif m_name == 'vgg16':
        model = vgg16(weights = VGG16_Weights.IMAGENET1K_V1)
        for name, param in model.named_parameters():
            if name.startswith('features.25') or name.startswith('features.26') or name.startswith('features.28') or name.startswith('classifier'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        model.classifier[-1] = nn.Linear(4096, num_classes) 
    elif m_name == 'mobilenet_v2':
        model  = mobilenet_v2(weights = MobileNet_V2_Weights.IMAGENET1K_V2)
        for name, param in model.named_parameters():
            if name.startswith('features.17') or name.startswith('features.18') or name.startswith('classifier'):
                param.requires_grad = True
            else:
                param.requires_grad = False
        model.classifier[-1] = nn.Linear(1280, num_classes)
    elif m_name == 'custom':
        model = CustomModel(num_classes)
    else:
        raise ValueError('Invalid model name')
    if best:
        model.load_state_dict(torch.load(f'results/{model_name}_synth{synth}_raw_noind_in_{"kd_" if knowledge_dist else ""}best.pth'))

    return model

def load_ensemble(synth, num_classes, device, best = False, knowledge_dist = False):
    models = [adjust_model(mn, num_classes) for mn in ['resnet18', 'vgg16', 'mobilenet_v2', 'custom']]
    mnames = ['resnet', 'vgg', 'mobilenet', 'custom']
    weights = [f'results/{mname}_synth{synth}_raw_noind_in_{"kd_" if knowledge_dist else ""}best.pth' for mname in mnames]
    #if any weights don't esixt, make them None
    for i, w in enumerate(weights):
        if not os.path.exists(w):
            print(f'Weight file {w} does not exist, setting to None')
            weights[i] = None
    ens_model = EnsembleModel(models, num_classes, device, weights)
    if best is True:
      ens_model.load_state_dict(torch.load(f'results/ensemble_synth{synth}_raw_noind_in_{"kd_" if knowledge_dist else ""}best.pth'))
    return ens_model

def get_neal_data(lab_file, test_path, classes):
    df = pd.read_csv(lab_file)
    df['common_name'] = df['class_label']
    df['begin_time'] = df['start_time'].astype('int')
    df['end_time'] = df['end_time'].astype('int')
    df_sub = df.loc[df['file_name'].isin(os.listdir('datasets/neal_data'))].copy()
    df_sub = df_sub.loc[df['confidence'] >= 0.9] 
    df_sub = df_sub.loc[df['labeler'].isin(["dk", "hh", "iw", "ms"])]
    df_sub = df_sub.loc[df_sub['common_name'].isin(classes)]
      # Convert start_time to integer
    df_sub = df_sub.drop_duplicates(subset=['file_name', 'begin_time'], keep='first')
    df_sub['formatted_file'] = df_sub.apply(lambda row: f'{row["file_name"]}_{int(row["begin_time"])}_{int(row["begin_time"])}.png', axis=1)

    # List all files in the directory
    files_in_directory = os.listdir(test_path)

    # Filter the DataFrame based on whether the formatted string is in the directory
    df_sub = df_sub[df_sub['formatted_file'].isin(files_in_directory)].copy()
    df_sub = df_sub.loc[df_sub['common_name'].isin(classes)]
    return df_sub    

def eval_model(model, loader, criterion, device, noisered, n_channel, denoise_net, metrics):
    running_loss = 0.0
    correct_test = 0
    total_test = 0
    precision_metric, recall_metric, f1_metric, accuracy_metric, top1_accuracy_metric, top5_accuracy_metric = metrics
    
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    accuracy_metric.reset()
    top1_accuracy_metric.reset()
    top5_accuracy_metric.reset()
    
    temperature = 3.0
    alpha = 0.7  # Weight for distillation loss
    
    with torch.no_grad():
        for inputs, labels in loader:
            embs = None
            if isinstance(labels, tuple) or isinstance(labels, list):
                embs = labels[1]
                labels = labels[0]
                embs = embs.to(device)
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = fast_resize_m1_1(inputs)
            #if noisered:
            #    inputs = spec_unet(inputs)[:,1:,:,:]
            if noisered:
                with torch.no_grad():
                    inputs = inputs - denoise_net(inputs)
                    inputs = fast_resize_m1_1(inputs)
            inputs = inputs.expand(-1, n_channel, -1, -1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if embs is not None:
              soft_targets = nn.functional.softmax(embs / temperature, dim=-1)
              soft_prob = nn.functional.log_softmax(outputs / temperature, dim=-1)

              # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
              distillation_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (temperature**2)
              loss = alpha * distillation_loss + (1 - alpha) * loss
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            if len(labels.shape) > 1:
                labels = torch.argmax(labels, dim=1)
            
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
             # Get the predicted class
            preds = torch.argmax(outputs, dim=1)
    
            # Update metrics
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            accuracy_metric.update(preds, labels)
            top1_accuracy_metric.update(outputs, labels)
            top5_accuracy_metric.update(outputs, labels)
    
    epoch_precision = precision_metric.compute()
    epoch_recall = recall_metric.compute()
    epoch_f1 = f1_metric.compute()
    epoch_accuracy = accuracy_metric.compute()
    epoch_top1 = 1.0 - top1_accuracy_metric.compute()
    epoch_top5 = 1.0 - top5_accuracy_metric.compute()
    
    test_loss = running_loss / len(loader)
    test_acc  = 100.0 * correct_test / total_test
    
    return test_loss, (epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, epoch_top1, epoch_top5)

def eval_model2(model, loader, criterion, device, noisered, n_channel, denoise_net, metrics):
    running_loss = 0.0
    correct_test = 0
    total_test = 0
    precision_metric, recall_metric, f1_metric, accuracy_metric, top1_accuracy_metric, top3_accuracy_metric, top5_accuracy_metric = metrics
    
    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    accuracy_metric.reset()
    top1_accuracy_metric.reset()
    top3_accuracy_metric.reset()
    top5_accuracy_metric.reset()
    
    all_y, all_preds = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = fast_resize_m1_1(inputs)
            #if noisered:
            #    inputs = spec_unet(inputs)[:,1:,:,:]
            if noisered:
                with torch.no_grad():
                    inputs = inputs - denoise_net(inputs)
                    inputs = fast_resize_m1_1(inputs)
            inputs = inputs.expand(-1, n_channel, -1, -1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
             # Get the predicted class
            preds = torch.argmax(outputs, dim=1)
            
            all_y.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
            # Update metrics
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            accuracy_metric.update(preds, labels)
            top1_accuracy_metric.update(outputs, labels)
            top3_accuracy_metric.update(outputs, labels)
            top5_accuracy_metric.update(outputs, labels)
    
    epoch_precision = precision_metric.compute()
    epoch_recall = recall_metric.compute()
    epoch_f1 = f1_metric.compute()
    epoch_accuracy = accuracy_metric.compute()
    epoch_top1 = 1.0 - top1_accuracy_metric.compute()
    epoch_top3 = 1.0 - top3_accuracy_metric.compute()
    epoch_top5 = 1.0 - top5_accuracy_metric.compute()
    
    test_loss = running_loss / len(loader)
    test_acc  = 100.0 * correct_test / total_test
    
    return test_loss, (epoch_accuracy, epoch_precision, epoch_recall, epoch_f1, epoch_top1, epoch_top3, epoch_top5), (all_y, all_preds)

def train_model(model, n_tuple, dfs, d_tuple, num_epochs = 10, save_interval = 10, batch_size=64):
    # Create an empty DataFrame to store results
    train_df_real, val_df, test_df, gen_path = dfs
    model_name, synthetic, noisered, aindex, mvn, knowledge_dist = n_tuple
    im_path, test_path, pop_classes, device, output_dir = d_tuple
    
    num_classes = len(pop_classes)

    is_cuda = device.type == 'cuda'
    
    if model_name == 'ensemble':
        n_channel = next(model.models[0].parameters()).size()[1]
    else:
        n_channel = next(model.parameters()).size()[1] # input channels
    
    if aindex:
        gen_path = gen_path.replace('spec_gan', 'spec_gan_aind')
        gen_path += '_aind'
    if mvn:
        gen_path += '_mvn'
    print("using", gen_path)
    gen_df = pd.DataFrame({'file_name': [os.path.join(gen_path, f) for f in os.listdir(gen_path)], 
                           'common_name': [f.split('_')[0] for f in os.listdir(gen_path)],
                           'begin_time': [0]*len(os.listdir(gen_path)), 
                           'end_time': [0]*len(os.listdir(gen_path))})
    gen_df = gen_df[gen_df['file_name'].apply(lambda x: int(x.split('_')[-1].split('.')[0])) < 250]
    
    if knowledge_dist:
      #train_df_emb = pd.read_csv('datasets/real_birdnet_emb.csv')
      train_df_emb = pd.read_csv('datasets/embeddings_all_table.csv')
      #replace any instance of '.wav' with '.png'
      train_df_emb['file_name'] = train_df_emb['file_name'].apply(lambda x: x.replace('.birdnet.embeddings.txt', ''))
      train_df_real = train_df_real.merge(train_df_emb, left_on=['file_name', 'begin_time'], right_on=['file_name', 'start_time'])
      val_df = val_df.merge(train_df_emb, left_on=['file_name', 'begin_time'], right_on=['file_name', 'start_time'])
      gen_df_emb = pd.read_csv('datasets/synth_birdnet_emb.csv')
      gen_df_emb['file_name'] = gen_df_emb['file_name'].apply(lambda x: x.replace('.birdnet.embeddings.txt.wav', '.png'))
      #add gen_path to gen_df_emb file_name
      gen_df_emb['file_name'] = gen_df_emb['file_name'].apply(lambda x: os.path.join(gen_path, x))
      gen_df = gen_df.merge(gen_df_emb, left_on=['file_name', 'begin_time'], right_on=['file_name', 'start_time'])
    
    ext = f'synth{str(synthetic)}_{"nr" if noisered else "raw"}_{"aind" if aindex else "noind"}_{"mvn" if mvn else "in"}{"_kd" if knowledge_dist else ""}'
    
    # Step 3: Sample from gen_df based on the same distribution
    if synthetic > 0:
        print(f"adding {synthetic} worth of synthetic data")
        train_distribution = train_df_real['common_name'].value_counts(normalize=True)
        gen_df_samp = pd.DataFrame(columns=gen_df.columns)
        for name, proportion in train_distribution.items():
            num_samples = synthetic#int(len(train_df_real) * proportion * synthetic)
    
            # Ensure we don't sample more synthetic rows than available for that species in gen_df
            available_samples = len(gen_df[gen_df['common_name'] == name])
            num_samples = min(num_samples, available_samples)
            
            # Sample without replacement
            if num_samples > 0:  # Check to avoid sampling 0 rows
                sampled_data = gen_df[gen_df['common_name'] == name].sample(n=num_samples, replace=False)
                gen_df_samp = pd.concat([gen_df_samp, sampled_data], axis=0)
        
        # Concatenate the real and synthetic data
        train_df = pd.concat([train_df_real, gen_df_samp], axis=0)
        print(f"Final training data has {len(train_df)} rows, with {len(gen_df_samp)} synthetic samples added.")
    else:
        train_df = train_df_real
        print(f"Final training data has {len(train_df)} rows, without synthetic samples added.")
      
    if noisered:
        nr_path = os.path.join('transformer_near', 'model_epoch_499.pth') # None
        denoise_net = ImageInpaintingModel()
        denoise_net.load_state_dict(torch.load(nr_path))
        denoise_net.to(device);
        denoise_net.eval();
        print("using", nr_path)
    else:
        denoise_net = None
    
    print(len(train_df), len(val_df), len(test_df))
    train_loader = DataLoader(dataset=SpecDatasetIm(train_df, im_path, pop_classes),  batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(dataset=SpecDatasetIm(val_df,   im_path, pop_classes),  batch_size=batch_size)
    #test_loader  = DataLoader(dataset=SpecDataset(test_df,  test_path, pop_classes), batch_size=batch_size)
    test_loader  = DataLoader(dataset=SpecDatasetIm(test_df,  test_path, pop_classes), batch_size=batch_size)
    results_df = pd.DataFrame(columns=['Name', 'Synthetic', 'Noisered', 'Epoch', 'Train Loss', 'Val Loss', 'Test Loss', 'Train Accuracy', 'Val Accuracy', 'Test Accuracy'])

    optimizer, criterion = Adam(model.parameters(), lr=0.001), nn.CrossEntropyLoss()
    
    temperature = 3.0
    alpha = 0.7  # Weight for distillation loss
    
    class EarlyStopping:
        def __init__(self, tolerance=3, min_delta=10):
    
            self.tolerance = tolerance
            self.min_delta = min_delta
            self.counter = 0
            self.early_stop = False
    
        def __call__(self, train_loss, validation_loss):
            if (validation_loss - train_loss) > self.min_delta:
                self.counter +=1
                if self.counter >= self.tolerance:  
                    self.early_stop = True
    
    early_stopping = EarlyStopping(tolerance=5, min_delta=10)
    #load_model = True
    #if load_model:
    #    results_df = pd.read_csv(os.path.join(output_dir, f'{model_name}_{ext}_training_results.csv'))
    #    load_path = '%s/%s_%s_epoch_%d.pth' % (output_dir, model_name, ext, 24)
    #    model.load_state_dict(torch.load(load_path))
    
    if is_cuda:
        model.to(device);
    
    precision_metric = torchmetrics.Precision(task="multiclass", average='macro', num_classes=num_classes).to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", average='macro', num_classes=num_classes).to(device)
    f1_metric = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=num_classes).to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    top1_accuracy_metric = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=num_classes).to(device)
    top5_accuracy_metric = torchmetrics.Accuracy(task="multiclass", top_k=5,num_classes=num_classes).to(device)
    
    best_val_accuracy = 0.0
    
    kd_print = True
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        precision_metric.reset()
        recall_metric.reset()
        f1_metric.reset()
        accuracy_metric.reset()
        top1_accuracy_metric.reset()
        top5_accuracy_metric.reset()
        
        print("Train")
        
        for inputs, labels in tqdm(train_loader):
            #if labels is tuple, split into labels and embs
            embs = None
            if isinstance(labels, tuple) or isinstance(labels, list):
                #import code; code.interact(local=dict(globals(), **locals()))
                embs = labels[1]
                labels = labels[0]
                embs = embs.to(device)
                if kd_print:
                  print("doing knowledge dist")
                  kd_print = False
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            #print(model, inputs, labels, device)
            #print(model.parameters().device, inputs.device, labels.device, device)
            optimizer.zero_grad()
            inputs = fast_resize_m1_1(inputs)
            if noisered:
                with torch.no_grad():
                    inputs = inputs - denoise_net(inputs)
                    inputs = fast_resize_m1_1(inputs)
                    #inputs = spec_unet(inputs)[:,1:,:,:]
            
            inputs = inputs.expand(-1, n_channel, -1, -1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            if embs is not None:
              soft_targets = nn.functional.softmax(embs / temperature, dim=-1)
              soft_prob = nn.functional.log_softmax(outputs / temperature, dim=-1)

              # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
              distillation_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (temperature**2)
              loss = alpha * distillation_loss + (1 - alpha) * loss
            
            loss.backward()
            optimizer.step()
            
            if len(labels.shape) > 1:
                labels = torch.argmax(labels, dim=1)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
             # Get the predicted class
            preds = torch.argmax(outputs, dim=1)
    
            # Update metrics
            precision_metric.update(preds, labels)
            recall_metric.update(preds, labels)
            f1_metric.update(preds, labels)
            accuracy_metric.update(preds, labels)
            top1_accuracy_metric.update(outputs, labels)
            top5_accuracy_metric.update(outputs, labels)
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100.0 * correct_train / total_train
        
        epoch_precision = precision_metric.compute()
        epoch_recall = recall_metric.compute()
        epoch_f1 = f1_metric.compute()
        epoch_accuracy = accuracy_metric.compute()
        epoch_top1 = 1.0 - top1_accuracy_metric.compute()
        epoch_top5 = 1.0 - top5_accuracy_metric.compute()
        
        # Testing
        model.eval()
        
        print("Val")
        val_loss, (val_accuracy, val_precision, val_recall, val_f1, val_top1, val_top5) = eval_model(model, val_loader, criterion, device, noisered, n_channel, denoise_net, (precision_metric, recall_metric, f1_metric, accuracy_metric, top1_accuracy_metric, top5_accuracy_metric))
        print("Test")
        test_loss, (test_accuracy, test_precision, test_recall, test_f1, test_top1, test_top5) = eval_model(model, test_loader, criterion, device, noisered, n_channel, denoise_net, (precision_metric, recall_metric, f1_metric, accuracy_metric, top1_accuracy_metric, top5_accuracy_metric))
        
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy  # update best accuracy
            best_model_path = f"{output_dir}/{model_name}_{ext}_best.pth"
            torch.save(model.state_dict(), best_model_path)  # save the model checkpoint
            print(f"New best model saved with Val Accuracy: {best_val_accuracy:.4f}")
        
        # Save results to the DataFrame
        if epoch % save_interval == 0:
            metrics = {
                'Name': [model_name],
                'Synthetic': [synthetic],
                'Noisered': [noisered],
                'Epoch': [epoch],
                'Train Loss': [train_loss],
                'Val Loss': [val_loss],
                'Test Loss': [test_loss],
                'Train Accuracy': [epoch_accuracy.item()],
                'Val Accuracy': [val_accuracy.item()],
                'Test Accuracy': [test_accuracy.item()],
                'Train Precision': [epoch_precision.item()],
                'Val Precision': [val_precision.item()],
                'Test Precision': [test_precision.item()],
                'Train Recall': [epoch_recall.item()],
                'Val Recall': [val_recall.item()],
                'Test Recall': [test_recall.item()],
                'Train F1': [epoch_f1.item()],
                'Val F1': [val_f1.item()],
                'Test F1': [test_f1.item()],
                'Train Top-1 Error': [epoch_top1.item()],
                'Val Top-1 Error': [val_top1.item()],
                'Test Top-1 Error': [test_top1.item()],
                'Train Top-5 Error': [epoch_top5.item()],
                'Val Top-5 Error': [val_top5.item()],
                'Test Top-5 Error': [test_top5.item()],
            }
            row_df = pd.DataFrame(metrics)
            if len(results_df) > 0:
                results_df = pd.concat([results_df, row_df], ignore_index=True)
            else: 
                results_df = row_df
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {100*val_accuracy:.2f}%, Test Loss: {test_loss:.4f}, Test Acc: {100*test_accuracy:.2f}%")
        if epoch % 10 == 0 or epoch == num_epochs - 1:

            # do checkpointing
            print("skipping saving model every 10 epochs for now")
            #torch.save(model.state_dict(), '%s/%s_%s_epoch_%d.pth' % (output_dir, model_name, ext, epoch))
        #clear_output(wait = True)
        # Save the final results to a CSV file
        results_df.to_csv(os.path.join(output_dir, f'{model_name}_{ext}_training_results.csv'), index=False)
        
        # early stopping
        #early_stopping(train_loss, val_loss)
        #if early_stopping.early_stop:
        #  print("Stopped at epoch:", i)
        #  break
    return

def eval_report_cm(split, model_name, ext, all_y, all_preds, num_classes, output_dir):
    
    list_labels = list(range(num_classes))
    cm = confusion_matrix(all_y, all_preds, labels = list_labels)
    
    report = classification_report(all_y, all_preds, output_dict=True, labels = list_labels, zero_division=0.0)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'{output_dir}/{model_name}_{ext}_classification_report.csv', index=True)
    
    # Plot the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    
    # Save the plot
    plt.savefig(f'{output_dir}/{model_name}_{ext}_{split}_confusion_matrix.png')
    plt.close()
    return

def eval_best_model(model, n_tuple, dfs, d_tuple, batch_size=64, knowledge_dist = False):
    model_name, synthetic, noisered, aindex, mvn = n_tuple
    val_df, test_df = dfs
    im_path, test_path, pop_classes, device, output_dir = d_tuple
    
    num_classes = len(pop_classes)
    
    is_cuda = device.type == 'cuda'
    
    if model_name == 'ensemble':
        n_channel = next(model.models[0].parameters()).size()[1]
    else:
        n_channel = next(model.parameters()).size()[1] # input channels
    
    ext = f'synth{str(synthetic)}_{"nr" if noisered else "raw"}_{"aind" if aindex else "noind"}_{"mvn" if mvn else "in"}{"kd_" if knowledge_dist else ""}'

    val_loader   = DataLoader(dataset=SpecDatasetIm(val_df, im_path, pop_classes),  batch_size=batch_size)
    test_loader  = DataLoader(dataset=SpecDatasetIm(test_df, test_path, pop_classes), batch_size=batch_size)
    results_df = pd.DataFrame(columns=['Name', 'Synthetic', 'Noisered', 'Epoch', 'Train Loss', 'Val Loss', 'Test Loss', 'Train Accuracy', 'Val Accuracy', 'Test Accuracy'])
    
    if is_cuda:
        model.to(device);
    model.eval()
    
    precision_metric = torchmetrics.Precision(task="multiclass", average='macro', num_classes=num_classes, zero_division=0).to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", average='macro', num_classes=num_classes, zero_division=0).to(device)
    f1_metric = torchmetrics.F1Score(task="multiclass", average='macro', num_classes=num_classes, zero_division=0).to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    top1_accuracy_metric = torchmetrics.Accuracy(task="multiclass", top_k=1, num_classes=num_classes).to(device)
    top3_accuracy_metric = torchmetrics.Accuracy(task="multiclass", top_k=3,num_classes=num_classes).to(device)
    top5_accuracy_metric = torchmetrics.Accuracy(task="multiclass", top_k=5,num_classes=num_classes).to(device)
    
    eval_metrics = precision_metric, recall_metric, f1_metric, accuracy_metric, top1_accuracy_metric, top3_accuracy_metric, top5_accuracy_metric
    
    criterion = nn.CrossEntropyLoss()
    
    denoise_net = None
    
    print("Val")
    val_loss, (val_accuracy, val_precision, val_recall, val_f1, val_top1, val_top3, val_top5), (val_y, val_preds) = eval_model2(model, val_loader, criterion, device, noisered, n_channel, denoise_net, eval_metrics)
    print("Test")
    test_loss, (test_accuracy, test_precision, test_recall, test_f1, test_top1, test_top3, test_top5), (test_y, test_preds) = eval_model2(model, test_loader, criterion, device, noisered, n_channel, denoise_net, eval_metrics)
    
    metrics = {
        'Name': [model_name],
        'Synthetic': [synthetic],
        'Noisered': [noisered],
        'Val Loss': [val_loss],
        'Test Loss': [test_loss],
        'Val Accuracy': [val_accuracy.item()],
        'Test Accuracy': [test_accuracy.item()],
        'Val Precision': [val_precision.item()],
        'Test Precision': [test_precision.item()],
        'Val Recall': [val_recall.item()],
        'Test Recall': [test_recall.item()],
        'Val F1': [val_f1.item()],
        'Test F1': [test_f1.item()],
        'Val Top-1 Error': [val_top1.item()],
        'Test Top-1 Error': [test_top1.item()],
        'Val Top-3 Error': [val_top3.item()],
        'Test Top-3 Error': [test_top3.item()],
        'Val Top-5 Error': [val_top5.item()],
        'Test Top-5 Error': [test_top5.item()],
    }
    row_df = pd.DataFrame(metrics)
    row_df.to_csv(os.path.join(output_dir, f'{model_name}_{ext}_eval_results.csv'), index=False)
    
    eval_report_cm("val", model_name, ext, val_y, val_preds, num_classes, output_dir)
    eval_report_cm("test", model_name, ext, test_y, test_preds, num_classes, output_dir)
    
    print(f'Num Synthetic: {synthetic}, Val accuracy: {val_accuracy.item():.4f}, Test accuracy: {test_accuracy.item():.4f}')
    
    

def train_specunet(spec_unet, train_loader, test_loader, optim, criterion, device, output_dir, num_epochs = 10, save_interval = 1):
    results_df = pd.DataFrame(columns=['epoch', 'train_loss', 'test_loss'])

    for epoch in range(num_epochs):

        spec_unet.train()
        running_loss = 0.0

        for i, data in tqdm(enumerate(train_loader, 0)):
            noise_batch, bird_batch, mix_batch, lab_batch = data
            
            optim.zero_grad()
            #wave_unet.zero_grad()
            
            sep_output = spec_unet(fast_resize_m1_1(mix_batch.to(device)))
            #noise_output, bird_output = torch.unsqueeze(sep_output[:,0,:], 1), torch.unsqueeze(sep_output[:,1,:], 1)

            #noise_output, bird_output = sep_output[:,0,:], sep_output[:,1,:]

            loss = criterion(sep_output, fast_resize_m1_1(torch.cat([noise_batch, bird_batch], dim = 1).to(device)))
            #bird_loss = bird_criterion(bird_output, bird_batch.to(device))

            #total_loss = noise_loss + bird_loss

            running_loss += loss.item()

            loss.backward()
            
            optim.step()
            #bird_optim.step()

            # compute the average loss
            curr_iter = epoch * len(train_loader) + i

            print('[%d/%d][%d/%d] Loss: %.4f'#| SDR: %.4f | FID: %.%4f' etc
                % (epoch, num_epochs, i, len(train_loader), loss.item()))
        
        train_loss = running_loss/len(train_loader)

        # Testing
        spec_unet.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for data in test_loader:
                noise_batch, bird_batch, mix_batch, lab_batch = data
                        
                sep_output = spec_unet(fast_resize_m1_1(mix_batch.to(device)))
                
                loss = criterion(sep_output, fast_resize_m1_1(torch.cat([noise_batch, bird_batch], dim = 1).to(device)))
            
                running_loss += loss.item()

        test_loss = running_loss / len(test_loader)
        
        plot_sep(noise_batch, bird_batch, mix_batch, 
                torch.unsqueeze(sep_output[:,0,:], 1).cpu(), torch.unsqueeze(sep_output[:,1,:], 1).cpu(), 
                output_dir, epoch, save = True);

        # Save results to the DataFrame
        if epoch % save_interval == 0:
            print('[%d/%d] TrainLoss:  %.4f | TestLoss: %.4f'#| SDR: %.4f'
                % (epoch, num_epochs, train_loss, test_loss))
            results_df = pd.concat([results_df, pd.DataFrame({'Epoch': [epoch], 'Train Loss': [train_loss], 'Test Loss': [test_loss]})],
                                ignore_index=True)
        

        if epoch % 10 == 0 or epoch == num_epochs - 1:

            # do checkpointing
            torch.save(spec_unet.state_dict(), '%s/specunet_epoch_%d.pth' % (output_dir, epoch))
        clear_output(wait = True)
        
        results_df.to_csv(os.path.join(output_dir, f'training_results.csv'), index=False)
    return 

def add_class_channels(x, y, n_C):
    zero_gen = torch.zeros(x.shape[:1] + (n_C,) + x.shape[-2:], dtype=torch.float32)
    mask = (y.view(-1, n_C, 1, 1) == 1).to(torch.float32)
    x_pad = zero_gen + mask
    return torch.cat([x, x_pad], dim=1)

def add_class_channels_cuda(x, y, n_C, device):
    zero_gen = torch.zeros(x.shape[:1] + (n_C,) + x.shape[-2:], dtype=torch.float32).to(device)
    mask = (y.view(-1, n_C, 1, 1) == 1).to(torch.float32).to(device)
    x_pad = zero_gen + mask
    return torch.cat([x, x_pad], dim=1).to(device)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

def plot_sep(noise_batch, bird_batch, mix_batch, noise_output, bird_output, output_dir, epoch, save = False):
    #plot spectrograms of noise_batch, bird_batch, mix_batch side by side
    nr = min(noise_batch.shape[0], 4)
    fig, ax = plt.subplots(nrows = nr, ncols=5, sharey=True)
    #noise_output, bird_output = torch.unsqueeze(noise_output, 1), torch.unsqueeze(bird_output, 1)
    
    for i in range(nr):
        ax[i,0].imshow(noise_batch.detach().numpy()[i][0,:,:])
        ax[i,1].imshow(bird_batch.detach().numpy()[i][0,:,:])
        ax[i,2].imshow(mix_batch.detach().numpy()[i][0,:,:])
        ax[i,3].imshow(noise_output.detach().numpy()[i][0,:,:])
        ax[i,4].imshow(bird_output.detach().numpy()[i][0,:,:])
        if i == 0:
            ax[i,0].set_title('Noise')
            ax[i,1].set_title('Bird')
            ax[i,2].set_title('Mix')
            ax[i,3].set_title('Noise Sep')
            ax[i,4].set_title('Bird Sep')
        if i != nr - 1:
            for j in range(5):
                ax[i,j].set_xticks([])

    #plt.show()
    if save:
        fig.savefig(os.path.join(f'%s/sep_examples_%03d.png' % (output_dir, epoch)))

def onehot(x, nc):
    return torch.nn.functional.one_hot(x, nc)
