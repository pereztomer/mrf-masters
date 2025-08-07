import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self, input_features=50, output_features=3, hidden_layers=[128, 256, 128]):
        super().__init__()

        layers = []
        in_features = input_features

        # Hidden layers
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(inplace=True)
            ])
            in_features = hidden_dim

        # Output layer (no activation - we'll add ReLU after)
        layers.append(nn.Linear(in_features, output_features))

        self.mlp = nn.Sequential(*layers)

    def bounded_output_layer(self, x):
        """Simple bounded output layer using ReLU and clipping."""
        # T1: Range 0.2-3.7
        t1 = torch.clamp(torch.relu(x[:, 0]), 0.2, 3.7)

        # T2: Range 0-0.8
        t2 = torch.clamp(torch.relu(x[:, 1]), 0.0, 0.8)

        # PD: Range 0-1
        pd = torch.clamp(torch.relu(x[:, 2]), 0.0, 1.0)

        return torch.stack([t1, t2, pd], dim=1)

    def forward(self, x):
        # Input: [batch_size, input_features] e.g., [1296, 50]
        x = self.mlp(x)
        # Apply sigmoid to each output individually (values between 0 and 1)
        x = self.bounded_output_layer(x)
        return x  # [batch_size, output_features] e.g., [1296, 3]


def create_simple_mlp(input_features, output_features, model_size="medium"):
    """
    Create simple MLP for time series processing

    Args:
        input_features: Number of input features (time steps)
        output_features: Number of output features (parameter maps)
        model_size: Model size - "tiny", "small", "medium", "large", "huge"
    """
    configs = {
        "tiny": {"hidden_layers": [64, 64]},
        "small": {"hidden_layers": [128, 128]},
        "medium": {"hidden_layers": [128, 256, 128]},
        "large": {"hidden_layers": [256, 512, 256]},
        "huge": {"hidden_layers": [512, 1024, 512, 256]},
        "huge+": {"hidden_layers": [256, 512, 1024, 2048, 512, 256]}
    }

    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")

    config = configs[model_size]
    return SimpleMLP(
        input_features=input_features,
        output_features=output_features,
        hidden_layers=config["hidden_layers"]
    )


# Usage example
if __name__ == "__main__":
    # Create simple MLP
    model = create_simple_mlp(
        input_features=50,  # Time steps
        output_features=3,  # T1, T2, PD
        model_size="huge"
    )

    # Test with batch of time series
    x = torch.randn(1296, 50)  # 1296 pixels, each with 50 time points
    y = model(x)
    print(f"Input: {x.shape} -> Output: {y.shape}")  # [1296, 50] -> [1296, 3]

    # Count parameters
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params / 1e6:.2f}M)")

    print(f"\nModel expects: [batch_size, {50}] -> outputs: [batch_size, {3}] (sigmoid normalized)")
    print("Each output (T1, T2, PD) individually normalized to [0, 1] range")
