import itertools
from math import factorial

import torch
from torch.utils.data import Dataset


def _get_tensor_size(a: torch.Tensor):
    return (a.element_size() * a.nelement()) / 2 ** 20


def mod_sum(x: torch.Tensor):
    return torch.fmod(torch.sum(x), 10)


def maj_vote(x: torch.Tensor):
    return torch.atleast_1d(torch.mode(x).values)


class VectorPVR(Dataset):
    def __init__(self,
                 size: int,
                 complexity: int = 0,
                 aggregation_method: str = 'mod_sum',
                 holdout: int = 0,
                 adversarial: bool = False):

        size = int(size)
        if size > 10 ** 7:
            raise ValueError('Size is to big! Ain`t nobody got RAM for that!')

        if not isinstance(complexity, int):
            raise ValueError(f"complexity must be an integer. Got {type(complexity).__name__} instead.")

        if complexity < 0 or complexity > 9:
            raise ValueError(f'complexity must be in [0,9]. Got complexity={complexity} instead.')

        if not isinstance(holdout, int):
            raise ValueError(f"holdout must be an integer. Got {type(holdout).__name__} instead.")

        if holdout < 0 or holdout > factorial(complexity + 1):
            raise ValueError(f"holdout must be in [0, {factorial(complexity + 1)}]. Got holdout={holdout} instead.")

        if complexity == 0 and holdout != 0:
            raise ValueError('Can`t holdout when complexity is 0.')

        self.size = size
        self.complexity = complexity
        self.aggregation_method = aggregation_method
        self.holdout = holdout
        self.adversarial = adversarial
        self.aggregator = self._set_up_aggregator(aggregation_method)

        self.data = torch.randint(10, size=(size, 11), dtype=torch.uint8)
        self.data[:, 0] = torch.randint(10 - complexity, size=[size], dtype=torch.uint8)

        if holdout > 0:
            if not adversarial:
                self._remove_permutations()
            else:
                self._insert_permutations()

    @staticmethod
    def _set_up_aggregator(aggregation_method):
        if aggregation_method == 'mod_sum':
            aggregator = mod_sum
        elif aggregation_method == 'min':
            aggregator = torch.min
        elif aggregation_method == 'max':
            aggregator = torch.max
        elif aggregation_method == 'median':
            aggregator = torch.median
        elif aggregation_method == 'maj_vote':
            aggregator = maj_vote
        else:
            raise ValueError('Unknown aggregation method.')

        return aggregator

    def _remove_permutations(self):
        # create # holdout permutation of (0, 1, ... , complexity)
        permutations = list(itertools.permutations(torch.arange(self.complexity + 1).tolist()))[:self.holdout]

        for permutation in permutations:
            sample_has_permutation = torch.ones(self.size, dtype=torch.bool)

            pointers = self.data[:, 0].long()
            for offset in range(self.complexity + 1):
                curr_value = self.data[range(self.size), pointers + 1 + offset]
                sample_has_permutation[curr_value != permutation[offset]] = False

            self.data[sample_has_permutation, 1:] = torch.fmod(self.data[sample_has_permutation, 1:], 10)

    def _insert_permutations(self):
        # create # holdout permutation of (0, 1, ... , complexity)
        permutations = list(itertools.permutations(torch.arange(self.complexity + 1).tolist()))[:self.holdout]
        permutations = torch.tensor(permutations, dtype=torch.uint8)
        permutations_selector = torch.randint(permutations.shape[0], size=[self.size])

        pointers = self.data[:, 0].long()
        for offset in range(self.complexity + 1):
            self.data[range(self.size), pointers + 1 + offset] = permutations[permutations_selector, offset]

    def __getitem__(self, idx):
        sample = self.data[idx]
        pointer = sample[0]
        value = self.aggregator(sample[pointer + 1: 1 + pointer + self.complexity + 1])

        return sample.long(), value.long()

    def __len__(self):
        return self.size


if __name__ == '__main__':
    p = 3
    ds = VectorPVR(10 ** p, complexity=2, aggregation_method='mod_sum', holdout=2, adversarial=True)
    print(p, ':', _get_tensor_size(ds.data))

    for idx in range(len(ds)):
        print(ds[idx])
