import torch


def quantize_int8(weights: torch.Tensor):
    scale = weights.abs().max() / 127
    quantized = torch.round(weights / scale).clamp(-127, 127).to(torch.int8)
    dequantized = quantized.float() * scale
    return dequantized, (weights - dequantized).abs()


typical_weights = (
    torch.randn(1000) * 0.1
)  # realistic small-magnitude weight distribution
with_outlier = typical_weights.clone()
with_outlier[0] = 5.0  # inject ONE outlier -- this is the whole experiment

_, error_normal = quantize_int8(typical_weights)
_, error_with_outlier = quantize_int8(with_outlier)

print(f"Mean error WITHOUT outlier: {error_normal.mean():.5f}")
print(
    f"Mean error WITH outlier (excluding the outlier itself): {error_with_outlier[1:].mean():.5f}"
)
print(
    f"That's a {error_with_outlier[1:].mean() / error_normal.mean():.1f}x degradation on weights that didn't change at all"
)
