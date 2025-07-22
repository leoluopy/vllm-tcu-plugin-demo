import torch

if __name__ == '__main__':

    cpu_pinned_zeros_explicit = torch.zeros(3, 4, device='cpu', pin_memory=True)

    print("END")
