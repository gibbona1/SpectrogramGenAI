import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class GeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GeneratorBlock, self).__init__()

        # First convolution block (before upsample)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        #second convolution block (once upsampled)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x_init = x  # Save input to add to output via skip connection
        # Apply the first convolution block

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Apply the second convolution block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # Add skip connection from input
        x = x + x_init

        # Upsample
        x = self.upsample(x)

        # Apply the third convolution block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        return x

class Generator(nn.Module):
    def __init__(self, noise_dim, output_channels, num_blocks):
        super(Generator, self).__init__()

        # Initial linear layer to map noise to the desired shape
        self.initial = nn.Linear(noise_dim, 512 * 16 * 16)

        # Create multiple generator blocks
        self.generator_blocks = nn.ModuleList()
        for i in range(num_blocks):
            in_c, out_c = 512//(2**i), 512//(2**(i+1))
            self.generator_blocks.append(GeneratorBlock(in_c, out_c))

        # Convolutional Block Attention Module (CBAM)
        self.cbam = ConvBlockAttentionModule(out_c)

        # Final convolutional layer
        self.final_conv = nn.Conv2d(out_c, output_channels, kernel_size=3, padding=1)

    def forward(self, noise):
        x = self.initial(noise)
        x = x.view(x.size(0), 512, 16, 16)  # Reshape to (batch_size, channels, height, width)

        #skip_connections = []  # Store skip connections for later use

        # Pass through generator blocks
        for block in self.generator_blocks:
            #print(x.shape)
            x = block(x)
        #    skip_connections.append(x)  # Store the skip connection
        #print(x.shape)
        #x = x +
        # Apply the Convolutional Block Attention Module
        x = self.cbam(x)

        # Final convolutional layer
        x = self.final_conv(x)

        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class ConvBlockAttentionModule(nn.Module):
    # Implement your CBAM module here
    def __init__(self, in_planes):
        super(ConvBlockAttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_planes)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorBlock, self).__init__()

        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1))
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, n_classes, n_blocks = 4):
        super(Discriminator, self).__init__()

        self.n_classes = n_classes

        #self.conv1 = nn.Conv2d(1+self.n_classes, 16, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)


        self.discriminator_blocks = nn.ModuleList()
        for i in range(n_blocks):
            in_c, out_c = (16*2**i), (16*2**(i + 1))
            self.discriminator_blocks.append(DiscriminatorBlock(in_c, out_c))

        # Layer 4
        self.layer_rf = nn.utils.spectral_norm(nn.Linear(256 * 16 * 16, 1)) #real/fake

        # Layer 5
        self.layer_c = nn.utils.spectral_norm(nn.Linear(256 * 16 * 16, self.n_classes)) #classification

        self.softmax = nn.LogSoftmax(dim = 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)

        for block in self.discriminator_blocks:
            x = block(x)
        # Flatten the feature map for fully connected layers
        x = x.view(x.size(0), -1)
        rf = self.sigmoid(self.layer_rf(x))
        c  = self.softmax(self.layer_c(x))

        return rf, c

class DiscriminatorInd(nn.Module):
    def __init__(self, n_classes, n_blocks = 4, n_ind = 11):
        super(DiscriminatorInd, self).__init__()

        self.n_classes = n_classes
        self.n_ind     = n_ind

        #self.conv1 = nn.Conv2d(1+self.n_classes, 16, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)


        self.discriminator_blocks = nn.ModuleList()
        for i in range(n_blocks):
            in_c, out_c = (16*2**i), (16*2**(i + 1))
            self.discriminator_blocks.append(DiscriminatorBlock(in_c, out_c))

        pen_c = 256 * 16 * 16 # penultimate layer size when flattened

        # Layer 4
        self.layer_rf = nn.utils.spectral_norm(nn.Linear(pen_c, 1)) #real/fake

        # Layer 5
        self.layer_c = nn.utils.spectral_norm(nn.Linear(pen_c, self.n_classes)) #classification

        self.layer_ai = nn.utils.spectral_norm(nn.Linear(pen_c, self.n_ind))

        self.softmax = nn.LogSoftmax(dim = 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv1(x)

        for block in self.discriminator_blocks:
            x = block(x)
        # Flatten the feature map for fully connected layers
        x = x.view(x.size(0), -1)
        # 3 outputs
        rf = self.sigmoid(self.layer_rf(x))
        c  = self.softmax(self.layer_c(x))
        a  = self.layer_ai(x)

        return rf, c, a

class ImageInpaintingModel(nn.Module):
  def __init__(self):
    super(ImageInpaintingModel, self).__init__()
    # CNN Downsample Block (ResNet18 for simplicity, without fully connected layer)
    #self.downsample = nn.Sequential(*list(resnet18(weights = ResNet18_Weights.IMAGENET1K_V1).children())[:-2])
    self.downsample = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(256, 512, kernel_size=3, stride=2),
        nn.ReLU(inplace=True)
        )

    # Transformer Block
    encoder_layers = TransformerEncoderLayer(d_model=512, nhead=8)
    self.transformer = TransformerEncoder(encoder_layer=encoder_layers, num_layers=6)

    # CNN Upsample Block
    self.upsample = nn.Sequential(
        nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True),
        nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.Tanh()
    )

  def forward(self, x):
    # Assuming x is of shape [B, C, H, W] -> [Batch Size, Channels, Height, Width]
    x = self.downsample(x)
    # Flatten and pass through Transformer
    b, c, h, w = x.shape
    xd = x.view(b, c, h * w).permute(2, 0, 1)  # Transformer expects [Seq Len, Batch, Features]
    x = self.transformer(xd)
    #x = x * xd # mulriplicative skip connection
    x = x.permute(1, 2, 0).view(b, c, h, w)

    # Upsample to original image size
    x = self.upsample(x)
    return x
