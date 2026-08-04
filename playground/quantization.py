import torch
from torch import Tensor

def quantize(x, num_bits=8):
    qmax = 2**(num_bits - 1) - 1
    scale = x.abs().max() / qmax
    x_q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return x_q, scale

def dequantize(x, scale):
    return x.to(torch.float32) * scale

def main():
    x = torch.randn(4, 4, dtype=torch.float32)
    qx, scale = quantize(x)
    dqx = dequantize(qx, scale)

    print(x)
    print(dqx)
    print((x - dqx).abs())
    print((x - dqx).abs().max())

    return 0

if __name__ == "__main__":
    main()