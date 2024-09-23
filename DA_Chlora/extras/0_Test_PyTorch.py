# %%
import torch

# Check if GPU is available
if torch.cuda.is_available():
    print("Number of GPUs available:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        device = torch.device("cuda:" + str(i))
        print("Device {}: {}".format(i, torch.cuda.get_device_name(device)))
        print("Memory Total: {:.2f} GB".format(torch.cuda.get_device_properties(device).total_memory / 1e9))
        print("Memory Free: {:.2f} GB".format(torch.cuda.memory_allocated(device) / 1e9))
else:
    print("No GPU available")
