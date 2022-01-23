import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

TRAIN_SET = datasets.MNIST(root='../data', train=True, download=True)
TEST_SET = datasets.MNIST(root='../data', train=False, download=True)

TRANSLATION_FACTOR = (40-28) / 2 / 40
TRANSFORM = transforms.Compose([
    transforms.CenterCrop([40, 40]),
    transforms.RandomAffine(0, translate=[TRANSLATION_FACTOR, TRANSLATION_FACTOR]),
    transforms.ToTensor()
])


def _get_value(labels):
    """
    get the value based on the pointer

    Args:
        labels: a tensor of length 4. where labels[0] is the pointer

    Returns:
        The value
    """

    pointer = labels[0]
    if 0 <= pointer <= 3:
        value = labels[1]
    elif 4 <= pointer <= 6:
        value = labels[2]
    else:
        value = labels[3]

    return value


class BlockStylePVR(Dataset):
    def __init__(self, train: bool):
        if train:
            self.ds = TRAIN_SET
        else:
            self.ds = TEST_SET

        labels = []
        idxs = []
        for idx, (_, y) in enumerate(self.ds):
            labels.append(y)
            idxs.append(idx)

        self.labels = torch.tensor(labels).reshape([-1, 4])
        self.idxs = torch.tensor(idxs).reshape([-1, 4])

    def __getitem__(self, idx):
        labels = self.labels[idx]
        idxs = self.idxs[idx]

        value = _get_value(labels)

        x = torch.zeros([1, 80, 80])
        x[0, :40, :40] = TRANSFORM(self.ds[idxs[0]][0])
        x[0, :40, 40:] = TRANSFORM(self.ds[idxs[1]][0])
        x[0, 40:, :40] = TRANSFORM(self.ds[idxs[2]][0])
        x[0, 40:, 40:] = TRANSFORM(self.ds[idxs[3]][0])

        return x, value

    def __len__(self):
        return self.idxs.shape[0]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds = BlockStylePVR(train=True)
    for i in range(50):
        plt.imshow(ds[0][0][0], cmap='gray')
        plt.show()
    print(ds[0][0].shape)
