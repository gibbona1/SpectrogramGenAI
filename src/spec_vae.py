import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

from diff_modules import VQAE, Decoder, Encoder, VQEmbeddingEMA
from diff_utils import get_data

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SCALE = 3
    WEIGHTS = [1, 4, 2, 1]
    WIN_SIZE = max(WEIGHTS)

    epochs = 10
    print_step = 50

    FORCE_BLANK_SILENCE = False

    batch_size = 128
    img_size = (256, 256)  # (width, height)

    input_dim = 1
    hidden_dim = 512
    latent_dim = 4
    n_embeddings = 512
    output_dim = 1
    commitment_beta = 0.25

    lr = 2e-4

    def plot_images_torch(original_images, z, z_quantised, reconstructed_images, sav_folder, epoch):
        # Ensure that the tensors are on the CPU and converted to numpy arrays
        original_images_np = original_images.cpu().detach().numpy()
        reconstructed_images_np = reconstructed_images.cpu().detach().numpy()
        z_np = z.cpu().detach().numpy()
        z_grid = np.concatenate(
            [
                np.concatenate([z_np[:, 0], z_np[:, 1]], axis=1),  # Concatenate first two channels horizontally
                np.concatenate([z_np[:, 2], z_np[:, 3]], axis=1),  # Concatenate next two channels horizontally
            ],
            axis=-1,
        )
        z_quantised_np = z_quantised.cpu().detach().numpy()
        z_quantised_grid = np.concatenate(
            [
                np.concatenate(
                    [z_quantised_np[:, 0], z_quantised_np[:, 1]], axis=1
                ),  # Concatenate first two channels horizontally
                np.concatenate(
                    [z_quantised_np[:, 2], z_quantised_np[:, 3]], axis=1
                ),  # Concatenate next two channels horizontally
            ],
            axis=-1,
        )
        # batch_size = original_images_np.shape[0]

        num_imgs = 5  # batch_size

        # Create a figure with a grid of subplots
        fig, axes = plt.subplots(4, num_imgs, figsize=(num_imgs * 4, 4))

        for i in range(num_imgs):  # batch_size):
            # Squeeze out the channel dimension and plot the original image on the top row
            axes[0, i].imshow(original_images_np[i, 0], cmap="viridis")
            axes[0, i].axis("off")
            axes[0, i].set_title(f"Original {i+1}")

            axes[1, i].imshow(z_grid[i], cmap="viridis")
            axes[1, i].axis("off")
            axes[1, i].set_title(f"Z Space {i+1}")

            axes[2, i].imshow(z_quantised_grid[i], cmap="viridis")
            axes[2, i].axis("off")
            axes[2, i].set_title(f"Z quantised {i+1}")

            # Squeeze out the channel dimension and plot the reconstructed image on the bottom row
            axes[3, i].imshow(reconstructed_images_np[i, 0], cmap="viridis")
            axes[3, i].axis("off")
            axes[3, i].set_title(f"Reconstruction {i+1}")

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        save_path = os.path.join(sav_folder, f"epoch_{epoch}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    config = SimpleNamespace(
        seed=42,
        batch_size=10,
        img_size=256,
        num_classes=27,
        dataset_path="Birdnet_conf_files_images_split",
        train_folder="train",
        val_folder="test",
        slice_size=1,
        do_validation=True,
        fp16=True,
        log_every_epoch=10,
        num_workers=10,
        lr=5e-3,
        load_model=False,
    )

    sav_folder = "vae_recon"
    model_sav_folder = os.path.join("models", "VQAE")
    if not os.path.exists(sav_folder):
        os.makedirs(sav_folder)
    if not os.path.exists(model_sav_folder):
        os.makedirs(model_sav_folder)

    train_loader, val_loader = get_data(config)

    encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=latent_dim)
    codebook = VQEmbeddingEMA(n_embeddings=n_embeddings, embedding_dim=latent_dim)
    decoder = Decoder(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    model = VQAE(Encoder=encoder, Codebook=codebook, Decoder=decoder).to(device)

    mse_loss = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Start training VQ-VAE...")
    for batch_idx, (x, _) in enumerate(val_loader):
        x = x.to(device)
        x_hat, z, z_quantized, _, _, _ = model(x)
        plot_images_torch(x, z, z_quantized, x_hat, sav_folder, 0)
        break
    model.train()

    for epoch in range(epochs):
        overall_loss = 0
        model.train()
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, _, _, commitment_loss, codebook_loss, perplexity = model(x)
            recon_loss = mse_loss(x_hat, x)

            loss = recon_loss + commitment_loss * commitment_beta + codebook_loss

            loss.backward()
            optimizer.step()

            if batch_idx % print_step == 0 or batch_idx == len(train_loader) - 1:
                print(
                    "epoch:",
                    epoch + 1,
                    "Train: (",
                    batch_idx + 1,
                    "/",
                    len(train_loader),
                    ") recon_loss:",
                    recon_loss.item(),
                    " perplexity: ",
                    perplexity.item(),
                    " commit_loss: ",
                    commitment_loss.item(),
                    "\n\t codebook loss: ",
                    codebook_loss.item(),
                    " total_loss: ",
                    loss.item(),
                    "\n",
                )
        model.eval()
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(val_loader):
            x = x.to(device)
            x_hat, z, z_quantized, commitment_loss, codebook_loss, perplexity = model(x)
            if batch_idx == 0:
                plot_images_torch(x, z, z_quantized, x_hat, sav_folder, epoch + 1)
            recon_loss = mse_loss(x_hat, x)
            loss = recon_loss + commitment_loss * commitment_beta + codebook_loss
            overall_loss += loss.item()
            if batch_idx % print_step == 0 or batch_idx == len(train_loader) - 1:
                print(
                    "epoch:",
                    epoch + 1,
                    "Val: (",
                    batch_idx + 1,
                    "/",
                    len(val_loader),
                    ") recon_loss:",
                    recon_loss.item(),
                    " perplexity: ",
                    perplexity.item(),
                    " commit_loss: ",
                    commitment_loss.item(),
                    "\n\t codebook loss: ",
                    codebook_loss.item(),
                    " total_loss: ",
                    loss.item(),
                    "\n",
                )

    # sve model
    torch.save(model.state_dict(), os.path.join(model_sav_folder, "ckpt.pt"))
    torch.save(optimizer.state_dict(), os.path.join(model_sav_folder, "optim.pt"))
    print("Finished!!")
