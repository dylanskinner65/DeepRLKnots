import torch

print(f"Number of devices: {torch.cuda.device_count()}\n")

for i in range(4):
    print(f"Name of devices: {torch.cuda.get_device_name(i)}")
