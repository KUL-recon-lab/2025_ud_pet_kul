from models import UNet3D


# count trainable parameters
def count_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


model = UNet3D(
    start_features=16,
    num_levels=4,
    down_conv=True,
    final_softplus=False,
)

print(f"Trainable parameters: {count_trainable(model):,}")
