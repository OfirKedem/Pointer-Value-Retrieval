import torch
from torch import nn
from datasets import VectorPVR

from torch.utils.data import DataLoader


class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(10, 64)
        self.l1 = nn.Linear(64 * VectorPVR.sample_size, 512)
        self.l2 = nn.Linear(512, 1024)
        self.l3 = nn.Linear(1024, 512)
        self.l4 = nn.Linear(512, 64)

        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = x.reshape(x.shape[0], -1)  # flatten embeddings (by batch)
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.relu(self.l3(x))
        x = torch.relu(self.l4(x))

        output = self.classifier(x)

        return output


def main():
    ds = VectorPVR(10)
    net = MLP()
    dl = DataLoader(ds, batch_size=2)

    for sample, value in dl:
        output = net(sample)
        print(output.shape)

        break


if __name__ == '__main__':
    main()
