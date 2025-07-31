import torch
from monai.bundle import download, ConfigParser

# Download the BraTS bundle
download(name="brats_mri_segmentation", bundle_dir="./bundles")

# Load the model using ConfigParser
bundle_root = "./bundles/brats_mri_segmentation"
parser = ConfigParser()
parser.read_config(f"{bundle_root}/configs/inference.json")

# Get the model
model = parser.get_parsed_content("network")
model.eval()

print(f"Model loaded: {type(model)}")
print(f"Model device: {next(model.parameters()).device}")


# Count total parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

# Count trainable parameters
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Get parameter counts
total_params = count_parameters(model)
trainable_params = count_trainable_parameters(model)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters (millions): {total_params/1e6:.2f}M")
print(f"Trainable parameters (millions): {trainable_params/1e6:.2f}M")
