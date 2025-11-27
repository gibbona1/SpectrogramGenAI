import numpy as np
from tqdm import tqdm
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from helpers import SpecDatasetIm, SpecDatasetImInd, fast_resize_m1_1
from gan_models import Generator, Discriminator, DiscriminatorInd, ImageInpaintingModel
import torch.optim as optim
import torchvision.utils as vutils
from IPython.display import clear_output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
is_cuda = device.type == 'cuda'

lab_path = 'model_output_loc_merge.csv'
ai_path = 'acoustic_indices.csv'
wav_path = 'Birdnet_conf_files'
im_path = 'Birdnet_conf_files_images'
specdata = np.load('specdata.npz', allow_pickle=True)

nr_path = os.path.join('transformer_near', 'model_epoch_499.pth') # None
use_aind = False
n_epochs = 200
st_epochs = 0

if nr_path:
    denoise_net = ImageInpaintingModel()
    denoise_net.load_state_dict(torch.load(nr_path))
    denoise_net.to(device);
    denoise_net.eval();

output_dir = f'spec_gan2'
if use_aind:
  output_dir = output_dir[:-1] + '_aind2'
if nr_path:
    output_dir += '_nr'
    print("Using Noise reduction model")
#make directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def normalize_cols(df, cols):
  for col in cols:
    df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=0)
  return df

df_cols = ['index0'] + list(pd.read_csv(lab_path).columns)
train_df, test_df = pd.DataFrame(specdata['train_df'], columns=df_cols), pd.DataFrame(specdata['test_df'], columns=df_cols)

if use_aind:
  aind_df = pd.read_csv(ai_path)
  a_indices = ["ACI", "ADiv", "AEve", "BioA", "H", "Ht", "M", "NDSI", "NDSIAnthro", "NDSIBio", "AR"]

  train_df = train_df.merge(aind_df,
                            left_on=['file_name', 'begin_time', 'end_time'],
                            right_on=['file_path', 'begin_time', 'end_time'])
  train_df = normalize_cols(train_df, a_indices)

  n_ind = len(a_indices)
else:
   n_ind = 11

pop_classes = specdata['categories']
num_classes = len(pop_classes)

batch_size = 16

#ensure that the batch size is a multiple of the length of the dataset
train_b = batch_size*(len(train_df)//batch_size)

if use_aind:
  train_loader = DataLoader(dataset=SpecDatasetImInd(train_df.iloc[:train_b], im_path, pop_classes, a_indices), batch_size=batch_size, shuffle=True)
else:
  train_loader = DataLoader(dataset=SpecDatasetIm(train_df.iloc[:train_b], im_path, pop_classes), batch_size=batch_size, shuffle=True)
#test_loader  = DataLoader(dataset=SpecDatasetIm(test_df.iloc[:test_b], im_path, pop_classes),  batch_size=batch_size)

N_Z        = 64
if use_aind:
  N_Z        = N_Z + n_ind
D          = 64
n_samp     = 128
BATCH_SIZE = batch_size

dis_label  = torch.FloatTensor((batch_size, 1))
aux_label  = torch.LongTensor((batch_size, 1))

eval_noise  = torch.randn(batch_size, N_Z + num_classes)
eval_label  = torch.randint(0, num_classes, (batch_size,))
eval_onehot = torch.zeros(batch_size, num_classes)
eval_onehot.scatter_(1, eval_label.unsqueeze(1), 1)
eval_noise[:, :num_classes] = eval_onehot

# Create the generator
output_channels = 1
num_blocks = 4

print("num_classes", num_classes)

netG = Generator(N_Z + num_classes, output_channels, num_blocks)
if use_aind:
  netD = DiscriminatorInd(num_classes, n_ind = n_ind)
else:
  netD = Discriminator(num_classes)

if st_epochs > 0:
    st_epochs = 10*(st_epochs//10)
    netG.load_state_dict(torch.load(os.path.join(output_dir, f'netG_epoch_{st_epochs}.pth')))
    netD.load_state_dict(torch.load(os.path.join(output_dir, f'netD_epoch_{st_epochs}.pth')))

lr = 0.002

# loss functions
BCE = nn.BCELoss()
NLL = nn.NLLLoss()
MSE = nn.MSELoss()

if is_cuda:
    netD.cuda()
    netG.cuda()
    BCE.cuda()
    NLL.cuda()
    MSE.cuda()
    dis_label, aux_label = dis_label.cuda(), aux_label.cuda()
    eval_noise = eval_noise.cuda()

FloatTensor = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if is_cuda else torch.LongTensor

gen_labels = torch.zeros(batch_size,).type(LongTensor).to(device)
z = torch.zeros(batch_size, N_Z+num_classes).type(FloatTensor).to(device)

# Adversarial ground truths
valid = torch.zeros((batch_size, 1)).type(FloatTensor).to(device).fill_(1.0)
fake  = torch.zeros((batch_size, 1)).type(FloatTensor).to(device).fill_(0.0)

real_imgs = torch.zeros(batch_size, 1, 256, 256).type(FloatTensor).to(device)
labels    = torch.zeros(batch_size,).type(LongTensor).to(device)
a_ind      = torch.zeros(batch_size, n_ind).type(FloatTensor).to(device)

# setup optimizer
optimizerD = optim.RMSprop(netD.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0) #default parameters (some optimization here?)
optimizerG = optim.RMSprop(netG.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0) #default parameters (some optimization here?)

res_path = os.path.join(output_dir, f'training_results.csv')
if st_epochs > 0:
  results_df = pd.read_csv(res_path)
else:
  results_df = pd.DataFrame(columns=['Epoch', 'Loss_D', 'Loss_G', 'Accuracy'])

for epoch in range(st_epochs + 1, n_epochs):
    all_loss_G, all_loss_D, all_loss_A = 0.0, 0.0, 0.0
    for i, data in tqdm(enumerate(train_loader, 0)):
        if use_aind:
          real_cpu, real_c, real_aind = data
        else:
          real_cpu, real_c = data
        #setup data
        if is_cuda:
            real_cpu = real_cpu.to(device)
            real_c   = real_c.to(device)
        real_cpu = fast_resize_m1_1(real_cpu)

        if nr_path:
          with torch.no_grad():
            #random_mask =
            real_cpu = real_cpu - denoise_net(real_cpu)
            real_cpu = fast_resize_m1_1(real_cpu)

        #print(real_cpu.shape)
        real_imgs.copy_(real_cpu)

        batch_size = real_cpu.size(0)

        # Configure input

        labels.copy_(real_c)
        if use_aind:
          a_ind.copy_(real_aind)

        # -----------------
        #  Train Generator
        # -----------------

        optimizerG.zero_grad()

        # Sample noise and labels as generator input
        noise = torch.randn(batch_size, N_Z + num_classes, device=device)
        # Generate random labels for fake classes
        fake_c = torch.randint(0, num_classes, (batch_size,), device=device)
        # Create one-hot encoded classes using PyTorch operations
        class_onehot = torch.zeros(batch_size, num_classes, device=device)
        class_onehot.scatter_(1, fake_c.unsqueeze(1), 1)

        # Modify the noise tensor with the one-hot encoded class information
        # Please note the correction in the slicing to ensure it works as intended
        # We also need to ensure we're modifying the correct portion of the noise tensor
        # Assuming you want to modify the first 'num_classes' elements of each noise vector
        noise[:, :num_classes] = class_onehot
        if use_aind:
          noise[:, num_classes:(num_classes + n_ind) ] = a_ind


        z.copy_(noise)
        gen_labels.copy_(fake_c)

        # Generate a batch of images
        gen_imgs = netG(z)

        # Loss measures generator's ability to fool the discriminator
        if use_aind:
          validity, pred_label, aind_output = netD(gen_imgs)
          loss_G = 0.33 * (BCE(validity, valid) + NLL(pred_label, gen_labels) + MSE(aind_output, a_ind))
        else:
          validity, pred_label = netD(gen_imgs)
          loss_G = 0.5 * (BCE(validity, valid) + NLL(pred_label, gen_labels))

        loss_G.backward()
        optimizerG.step()



        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizerD.zero_grad()

        # Loss for real images
        if use_aind:
          real_pred, real_aux, aind_output = netD(real_imgs)
          d_real_loss = 0.33 * (BCE(real_pred, valid) + NLL(real_aux, labels) + MSE(aind_output, a_ind))
        else:
          real_pred, real_aux = netD(real_imgs)
          d_real_loss = (BCE(real_pred, valid) + NLL(real_aux, labels)) / 2

        # Loss for fake images
        if use_aind:
          fake_pred, fake_aux, aind_output = netD(gen_imgs.detach())
          d_fake_loss = 0.33 * (BCE(fake_pred, fake) + NLL(fake_aux, gen_labels) + MSE(aind_output, a_ind))
        else:
          fake_pred, fake_aux = netD(gen_imgs.detach())
          #print(fake_pred.shape, fake.shape, fake_aux.shape, gen_labels.shape)
          d_fake_loss = (BCE(fake_pred, fake) + NLL(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        loss_D = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        loss_D.backward()
        optimizerD.step()

        # Compute metrics
        #with torch.no_grad():
        all_loss_D += loss_D.item()
        all_loss_G += loss_G.item()
        all_loss_A += d_acc

        if i == 0 and epoch == 1:
          vutils.save_image(real_cpu, '%s/real_samples.png' % output_dir)

        if i % 100 == 0 or i == len(train_loader) - 1:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Acc: %.4f'
            % (epoch, n_epochs, i, len(train_loader),
                loss_D.item(), loss_G.item(), d_acc))
            print('Label for eval = {}'.format(eval_label))
            eval_imgs = netG(eval_noise.squeeze())
            vutils.save_image(
                eval_imgs.data,
                '%s/fake_samples_epoch_%03d.png' % (output_dir, epoch)
            )
    # Save results to the DataFrame
    row_df = pd.DataFrame({'Epoch': [epoch],
                           'Loss_D': [all_loss_D/len(train_loader)],
                           'Loss_G': [all_loss_G/len(train_loader)],
                           'Loss_A': [all_loss_A/len(train_loader)]})
    if len(results_df) > 0:
        results_df = pd.concat([results_df, row_df], ignore_index=True)
    else:
        results_df = row_df
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (output_dir, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (output_dir, epoch))
    clear_output(wait = True)
    results_df.to_csv(res_path, index=False)
