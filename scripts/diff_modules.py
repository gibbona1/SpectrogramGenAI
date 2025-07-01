
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from diff_utils import *
import logging
import wandb
from fastprogress import progress_bar


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)


        if remove_deep_conv:
            self.bot1 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 256)
        else:
            self.bot1 = DoubleConv(256, 512)
            self.bot2 = DoubleConv(512, 512)
            self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t)


class UNet_conditional(UNet):
    def __init__(self, c_in=1, c_out=1, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        return self.unet_forwad(x, t)


class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size=(4, 4, 3, 1), stride=2):
        super(Encoder, self).__init__()
        
        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_size
        
        self.strided_conv_1 = nn.Conv2d(input_dim, hidden_dim, kernel_1, stride, padding=1)
        self.strided_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, stride, padding=1)
        
        self.residual_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_3, padding=1)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_4, padding=0)
        
        self.proj = nn.Conv2d(hidden_dim, output_dim, kernel_size=1)
        
    def forward(self, x):
        
        x = self.strided_conv_1(x)
        x = self.strided_conv_2(x)
        
        x = F.relu(x)
        y = self.residual_conv_1(x)
        y = y+x
        
        x = F.relu(y)
        y = self.residual_conv_2(x)
        y = y+x
        
        y = self.proj(y)
        return y

class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon
        
        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)

        distances = (-torch.cdist(x_flat, self.embedding, p=2)) ** 2

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices.view(x.size(0), x.size(1))
    
    def retrieve_random_codebook(self, random_indices):
        quantized = F.embedding(random_indices, self.embedding)
        quantized = quantized.transpose(1, 3)
        
        return quantized

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D)
        
        distances = (-torch.cdist(x_flat, self.embedding, p=2)) ** 2

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        
        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)
            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw
            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        codebook_loss = F.mse_loss(x.detach(), quantized)
        e_latent_loss = F.mse_loss(x, quantized.detach())
        commitment_loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, commitment_loss, codebook_loss, perplexity
    
class Decoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_sizes=(1, 3, 2, 2), stride=2):
        super(Decoder, self).__init__()
        
        kernel_1, kernel_2, kernel_3, kernel_4 = kernel_sizes
        
        self.in_proj = nn.Conv2d(input_dim, hidden_dim, kernel_size=1)
        
        self.residual_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, kernel_1, padding=0)
        self.residual_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_2, padding=1)
        
        self.strided_t_conv_1 = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_3, stride, padding=0)
        self.strided_t_conv_2 = nn.ConvTranspose2d(hidden_dim, output_dim, kernel_4, stride, padding=0)
        
    def forward(self, x):

        x = self.in_proj(x)
        
        y = self.residual_conv_1(x)
        y = y+x
        x = F.relu(y)
        
        y = self.residual_conv_2(x)
        y = y+x
        y = F.relu(y)
        
        y = self.strided_t_conv_1(y)
        y = self.strided_t_conv_2(y)
        
        return y

class VQAE(nn.Module):
    def __init__(self, Encoder, Codebook, Decoder):
        super(VQAE, self).__init__()
        self.encoder = Encoder
        self.codebook = Codebook
        self.decoder = Decoder
                
    def forward(self, x):
        z = self.encoder(x)
        z_quantized, commitment_loss, codebook_loss, perplexity = self.codebook(z)
        x_hat = self.decoder(z_quantized)
        
        return x_hat, z, z_quantized, commitment_loss, codebook_loss, perplexity

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=1, c_out=1, device=device, **kwargs):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(c_in, c_out, num_classes=num_classes, **kwargs).to(device)
        #self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        n = len(labels)
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        #self.ema.step_ema(self.ema_model, self.model)
        #self.scheduler.step()
    
    def fast_resize_m1_1(self, x):
        min_values = x.reshape(x.shape[0],-1).min(dim=-1,keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        max_values = x.reshape(x.shape[0],-1).max(dim=-1,keepdim=True)[0].unsqueeze(2).unsqueeze(3)
        m = max_values - min_values
        x = (x - min_values)/m
        x = (1 * (m >= 0) - 1 * (m < 0)) * 2 * (x - 0.5)
        return x

    def one_epoch(self, train=True):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda", dtype=torch.float16) and (torch.inference_mode() if not train else torch.enable_grad()):
                images = self.fast_resize_m1_1(images).to(self.device)
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                images, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                images = self.model(images, t, labels)
                loss = self.mse(noise, images)
                avg_loss += loss
                if train:
                    self.train_step(loss)
                    wandb.log({"train_mse": loss.item(),
                                "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"MSE={loss.item():2.3f}"        
        return avg_loss.mean().item()

    def log_images(self):
        "Log images to wandb and save them to disk"
        labels = torch.arange(self.num_classes).long().to(self.device)
        #do this in 3 steps and append it all to sampled_images
        sampled_images = []
        for i in range(0, self.num_classes, 5):
            sampled_images.append(self.sample(use_ema=False, labels=labels[i:i+5]))
        sampled_images = torch.vstack(sampled_images)
        #sampled_images = self.sample(use_ema=False, labels=labels)
        #import code; code.interact(local=dict(globals(), **locals()))
        wandb.log({"sampled_images": [
            wandb.Image(plt.cm.viridis(img.permute(1, 2, 0).cpu().numpy().squeeze()))
            for img in sampled_images]})

        # EMA model sampling
        #ema_sampled_images = self.sample(use_ema=True, labels=labels)
        #plot_images(sampled_images)  #to display on jupyter if available
        #wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_sampled_images]})

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        #self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch=-1):
        "Save model locally and on wandb"
        torch.save(self.model.state_dict(), os.path.join("models", run_name, f"ckpt.pt"))
        #torch.save(self.ema_model.state_dict(), os.path.join("models", run_name, f"ema_ckpt.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("models", run_name, f"optim.pt"))
        at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional", metadata={"epoch": epoch})
        at.add_dir(os.path.join("models", run_name))
        wandb.log_artifact(at)
    
    def load_model(self, args):
        # Load model state
        if not args.load_model:
            print("Starting model ftesh...")
            return
        
        model_path = os.path.join("models", args.run_name, f"ckpt.pt")
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path))
            print(f"Model loaded successfully from {model_path}")
        else:
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
        # Load optimizer state
        optim_path = os.path.join("models", args.run_name, f"optim.pt")
        if os.path.exists(optim_path):
            self.optimizer.load_state_dict(torch.load(optim_path))
            print(f"Optimizer state loaded successfully from {optim_path}")
        else:
            raise FileNotFoundError(f"Optimizer checkpoint not found at {optim_path}")
        
        print("Model loaded successfully")

    def prepare(self, args):
        mk_folders(args.run_name)
        self.train_dataloader, self.val_dataloader = get_data(args)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=True):
            logging.info(f"Starting epoch {epoch}:")
            _  = self.one_epoch(train=True)
            
            ## validation
            if args.do_validation:
                avg_loss = self.one_epoch(train=False)
                wandb.log({"val_mse": avg_loss})
            
            # log predicitons
            if epoch % args.log_every_epoch == 0 or epoch == args.epochs - 1:
                self.log_images()

        # save model
        self.save_model(run_name=args.run_name, epoch=epoch)

class DiffusionVAE(Diffusion):
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=1, c_out=1, device=device, vqae_path = 'models/VQAE/ckpt.pt', sav_denoise_path = None, class_names = [], **kwargs):
        super().__init__(noise_steps, beta_start, beta_end, img_size, num_classes, c_in, c_out, device, **kwargs)
        
        input_dim = 1
        hidden_dim = 512
        latent_dim = 4
        n_embeddings= 512
        output_dim = 1
        
        encoder = Encoder(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=latent_dim)
        codebook = VQEmbeddingEMA(n_embeddings=n_embeddings, embedding_dim=latent_dim)
        decoder = Decoder(input_dim=latent_dim, hidden_dim=hidden_dim, output_dim=output_dim)
        self.vqae = VQAE(Encoder=encoder, Codebook=codebook, Decoder=decoder).to(device)
        self.vqae.load_state_dict(torch.load(vqae_path, map_location=device))
        self.vqae.eval()
        
        self.img_size = img_size//4
        self.model = UNet_conditional(latent_dim, latent_dim, num_classes=num_classes, **kwargs).to(device)
        
        self.c_in = latent_dim
        
        self.sav_denoise_path = sav_denoise_path
        self.class_names = class_names
        #self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)

    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        vqae = self.vqae
        n = len(labels)
        logging.info(f"Sampling {n} new images....")
        model.eval()
        vqae.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in progress_bar(reversed(range(1, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                if self.sav_denoise_path:
                    if i % 50 == 0 or i == 1 or i == self.noise_steps-1:
                        print(f'saving denoise at step {i}...')
                        x_denoise = x
                        x_denoise = x_denoise.clamp(-1, 1)
                        
                        x_denoise, _, _, _ = self.vqae.codebook(x_denoise)
                        x_denoise_up = vqae.decoder(x_denoise)
                        
                        x_denoise_up = (x_denoise_up + 1) / 2
                        x_denoise_up = (x_denoise_up * 255).type(torch.uint8)
                        x_denoise = (x_denoise + 1) / 2
                        x_denoise = (x_denoise * 255).type(torch.uint8)
                        for img_i, img_i_up, lab_i in zip(x_denoise, x_denoise_up, labels):
                            img_i_grid = torch.cat([
                                torch.cat([img_i[0], img_i[1]], dim=1),  # Concatenate first two channels horizontally
                                torch.cat([img_i[2], img_i[3]], dim=1)   # Concatenate next two channels horizontally
                            ], dim=0)  # Concatenate the two rows vertically
                            
                            # Convert the tensor to a numpy array and map it to viridis color map
                            img_i_grid = plt.cm.viridis(img_i_grid.cpu().numpy() / 255.0)
                            img_i_grid = (img_i_grid * 255).astype(np.uint8)
                        
                            # Convert the numpy array to an image
                            img_i_grid = Image.fromarray(img_i_grid)
                        
                            # Save the image
                            img_i_grid.save(f"{self.sav_denoise_path}/{self.class_names[lab_i]}_noise_{i}_latent.png")
                            
                            img_i_up = plt.cm.viridis(img_i_up.permute(1, 2, 0).cpu().numpy().squeeze())
                            img_i_up = (img_i_up * 255).astype(np.uint8)
                            img_i_up = Image.fromarray(img_i_up)
                            img_i_up.save(f"{self.sav_denoise_path}/{self.class_names[lab_i]}_noise_{i}_decode.png")
                            
                            
        x = x.clamp(-1, 1)
        x, _, _, _ = self.vqae.codebook(x)
        x = vqae.decoder(x)
        x = (x + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def one_epoch(self, train=True):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda", dtype=torch.float16) and (torch.inference_mode() if not train else torch.enable_grad()):
                images = self.vqae.encoder(self.fast_resize_m1_1(images).to(self.device))
                labels = labels.to(self.device)
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                images, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                images = self.model(images, t, labels)
                #import code; code.interact(local=dict(globals(), **locals()))
                loss = self.mse(noise, images)
                avg_loss += loss
                if train:
                    self.train_step(loss)
                    wandb.log({"train_mse": loss.item(),
                                "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"MSE={loss.item():2.3f}"        
        return avg_loss.mean().item()

    def log_images(self):
        "Log images to wandb and save them to disk"
        labels = torch.arange(self.num_classes).long().to(self.device)
        #do this in 3 steps and append it all to sampled_images
        #sampled_images = []
        #for i in range(0, self.num_classes, 5):
        #    sampled_images.append(self.sample(use_ema=False, labels=labels[i:i+5]))
        #sampled_images = torch.vstack(sampled_images)
        sampled_images = self.sample(use_ema=False, labels=labels)
        #import code; code.interact(local=dict(globals(), **locals()))
        wandb.log({"sampled_images": [
            wandb.Image(plt.cm.viridis(img.permute(1, 2, 0).cpu().numpy().squeeze()))
            for img in sampled_images]})

        # EMA model sampling
        #ema_sampled_images = self.sample(use_ema=True, labels=labels)
        #plot_images(sampled_images)  #to display on jupyter if available
        #wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_sampled_images]})
    
    def gen_images(self, img_folder, samp_i, labels=None):
        "Log images to wandb and save them to disk"
        class_names = self.class_names
        if labels is None:
            labels = torch.arange(self.num_classes).long().to(self.device)
        #do this in 3 steps and append it all to sampled_images
        #sampled_images = []
        #for i in range(0, self.num_classes, 5):
        #    sampled_images.append(self.sample(use_ema=False, labels=labels[i:i+5]))
        #sampled_images = torch.vstack(sampled_images)
        sampled_images = self.sample(use_ema=False, labels=labels)
        
        if self.sav_denoise_path:
            print("not saving image, just noise portions")
            return
        #import code; code.interact(local=dict(globals(), **locals()))
        #save images to config.img_folder
        for i, (lab, img) in enumerate(zip(labels, sampled_images)):
            img = plt.cm.viridis(img.permute(1, 2, 0).cpu().numpy().squeeze())
            img = (img * 255).astype(np.uint8)
            img = Image.fromarray(img)
            img.save(f"{img_folder}/{class_names[lab]}_gen_imgs_{i}_{samp_i}.png")

