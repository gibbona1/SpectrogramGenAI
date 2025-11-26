import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics.image.inception import InceptionScore
from PIL import Image
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, file) for file in os.listdir(root_dir) if file.endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')  # Convert image to RGB (for Inception)
        if self.transform:
            image = self.transform(image)
        return image  # We don't return any labels

# Function to compute inception score
def compute_inception_score(image_folder, batch_size=32):
  # Define image transformations to resize images to 299x299 (input size for Inception model)
  transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255),  # Scale to match uint8 range
  ])

  # Create a dataset from the image folder
  dataset = ImageDataset(image_folder, transform=transform)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
  
  # Initialize Inception Score metric
  inception = InceptionScore()
  
  # Iterate over the dataset and update inception metric
  for imgs in tqdm(dataloader):
    #imgs, _ = batch  # We ignore the labels since we're only interested in images
    imgs = imgs.type(torch.uint8)  # Ensure data type is uint8
    inception.update(imgs)
  
  # Compute Inception Score
  score = inception.compute()
  print(f'Inception Score: {score[0].item()} ± {score[1].item()}')

if __name__ == "__main__":
  # Check if image folder argument is provided
  if len(sys.argv) != 2:
    print("Usage: python compute_inception_score.py <image_folder>")
    sys.exit(1)
  
  # Get the image folder from the command-line argument
  image_folder = sys.argv[1]
  
  # Check if folder exists
  if not os.path.isdir(image_folder):
    print(f"Error: Folder '{image_folder}' not found.")
    sys.exit(1)
  
  print(f"Computing Inception Score for images in '{image_folder}'")
  # Compute Inception Score with a batch size of 32 (can adjust if needed)
  compute_inception_score(image_folder, batch_size=32)
