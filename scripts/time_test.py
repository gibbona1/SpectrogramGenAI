import torch
import time
from helpers import load_ensemble, adjust_model

print(torch.__version__)

input_data = torch.randn(16, 3, 256, 256) # Adjust input shape
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = load_ensemble(0, 27, device, knowledge_dist=False)
model = adjust_model('resnet', 27)
compiled_model = torch.compile(model, mode='reduce-overhead')
compiled_model = compiled_model.to(device)
compiled_model.eval()
input_data = input_data.to(device)

# Warm-up run (to account for initialization overhead)
for _ in range(5):
    _ = compiled_model(input_data)

# Measure time
start_time = time.time()
with torch.no_grad():
    output = compiled_model(input_data)
end_time = time.time()

print(f"Ensemble Forward Pass Time: {(end_time - start_time) * 1000:.2f} ms")
