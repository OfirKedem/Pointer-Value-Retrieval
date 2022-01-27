import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

TRAIN_SET = datasets.MNIST(root='../data', train=True, download=True)
TEST_SET = datasets.MNIST(root='../data', train=False, download=True)

HOLDOUT_CLASSES = {"top_left": [],
                   "top_right": [1, 2, 3],
                   "bottom_left": [4, 5, 6],
                   "bottom_right": [7, 8, 9, 0]}

TRANSLATION_FACTOR = (40 - 28) / 2 / 40
TRANSFORM = transforms.Compose([
    transforms.CenterCrop([40, 40]),  # basically pads with zeros
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


def _get_holdout_idxs(label2idxs: dict):
    ds_size = sum([len(idxs) for idxs in label2idxs.values()])
    pvr_ds_size = ds_size // 4
    per_block_per_label_size = pvr_ds_size // 10

    tl_idxs = torch.concat([torch.tensor(idxs)[:per_block_per_label_size] for idxs in label2idxs.values()])

    bl_idxs = []
    br_idxs = []


class BlockStylePVR(Dataset):
    def __init__(self,
                 train: bool,
                 mode: str = "iid",
                 size: int = None):
        """

        Args:
            train: whether to use MNIST train set, else use the test set.
            mode: "holdout" or "adversarial" or "iid".
            size: dataset size
        """
        if train:
            self.ds = TRAIN_SET
        else:
            self.ds = TEST_SET

        if size is not None and size > len(self.ds) // 4:
            raise ValueError(f"Requested dataset size is too big. Can be up too {len(self.ds) // 4}.")

        self.pvr_ds_size = len(self.ds) // 4 if size is None else int(size)

        labels = []
        for idx, (_, y) in enumerate(self.ds):
            labels.append(y)
        labels = torch.tensor(labels)

        self.idxs = torch.zeros([self.pvr_ds_size, 4], dtype=torch.long)
        self.labels = torch.zeros([self.pvr_ds_size, 4], dtype=torch.long)

        if mode == 'iid':
            self.labels = labels.reshape([-1, 4])
            self.labels = self.labels[:self.pvr_ds_size]

            self.idxs = torch.arange(len(self.ds)).reshape([-1, 4])
            self.idxs = self.idxs[:self.pvr_ds_size]

        elif mode == "holdout":
            for i, holdout_class in enumerate(HOLDOUT_CLASSES.values()):
                probs = torch.ones(len(self.ds))
                for label in holdout_class:
                    probs[labels == label] = 0
                curr_idxs = torch.multinomial(probs, self.pvr_ds_size)
                self.idxs[:, i] = curr_idxs
                self.labels[:, i] = labels[curr_idxs]

        elif mode == "adversarial":
            for i, holdout_class in enumerate(HOLDOUT_CLASSES.values()):
                probs = torch.ones(len(self.ds)) if i == 0 else torch.zeros(len(self.ds))
                for label in holdout_class:
                    probs[labels == label] = 1
                curr_idxs = torch.multinomial(probs, self.pvr_ds_size)
                self.idxs[:, i] = curr_idxs
                self.labels[:, i] = labels[curr_idxs]

        else:
            raise ValueError("Unknown dataset mode.")

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
        return self.pvr_ds_size


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    ds = BlockStylePVR(train=True, mode='iid', size=None)
    print(len(ds))
    for i in range(50):
        x, y = ds[i]
        plt.imshow(x[0], cmap='gray')
        plt.title(y.item())
        plt.show()

    print(ds[0][0].shape)
