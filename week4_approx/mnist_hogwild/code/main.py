from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torchvision import datasets, transforms

def test_mp():
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=10, shuffle=True, num_workers=1)
    conv_1 = nn.Conv2d(1, 10, kernel_size=5)
    for batch_idx, (data, target) in enumerate(train_loader):
        print(data.shape)
        print(conv_1(data).shape)

if __name__ == '__main__':
    p = mp.Process(target=test_mp)
    # We first train the model across `num_processes` processes
    p.start()
    p.join()