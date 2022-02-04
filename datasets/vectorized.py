import itertools
from math import factorial

import torch
from torch.utils.data import Dataset


# calculate vector size in MB
def _get_tensor_size(a: torch.Tensor):
    return (a.element_size() * a.nelement()) / 2 ** 20


def mod_sum(x: torch.Tensor):
    return torch.fmod(torch.sum(x), 10)


def maj_vote(x: torch.Tensor):
    return torch.atleast_1d(torch.mode(x).values)


class VectorPVR(Dataset):
    sample_size = 11  # pointer + 10 digits

    def __init__(self,
                 size: int,
                 complexity: int = 0,  # size of the window (m in the article)
                 holdout: int = 0,  # number of permutations to holdout ('holdout-i' or '# holdout' in article)
                 aggregation_method: str = 'mod_sum',
                 adversarial: bool = False):

        size = int(size)
        if size > 10 ** 8:
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

        # data is built from numbers 0-9
        self.data = torch.randint(10, size=(size, self.sample_size), dtype=torch.uint8)

        # the pointer value should be limited so the window will fit in the samples
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

    def get_holdout_permutations(self):
        # find the first 'holdout' permutations of (0, 1, ... , complexity)
        permutations_iterator = itertools.permutations(range(self.complexity + 1))
        return [next(permutations_iterator) for i in range(self.holdout)]

    def _remove_permutations(self):
        # create # holdout permutation of (0, 1, ... , complexity)
        permutations = self.get_holdout_permutations()

        for permutation in permutations:
            # indicator of where the permutation was found (default true)
            sample_has_permutation = torch.ones(self.size, dtype=torch.bool)

            # compare the value window to the permutation
            # if there's a mismatch in a single index mark as False
            pointers = self.data[:, 0].long()
            for offset in range(self.complexity + 1):
                curr_value = self.data[range(self.size), pointers + 1 + offset]
                sample_has_permutation[curr_value != permutation[offset]] = False

            # add 1 (mod 10) to the vectors where the permutation was found (excluding the pointer at index 0)
            self.data[sample_has_permutation, 1:] = torch.fmod(self.data[sample_has_permutation, 1:] + 1, 10)

    def _insert_permutations(self):
        # create # holdout permutation of (0, 1, ... , complexity)
        permutations = torch.tensor(self.get_holdout_permutations(), dtype=torch.uint8)
        permutations_selector = torch.randint(permutations.shape[0], size=[self.size])
        # permutations_selector = torch.zeros(size=[self.size])

        pointers = self.data[:, 0].long()
        for offset in range(self.complexity + 1):
            self.data[range(self.size), pointers + 1 + offset] = permutations[permutations_selector, offset]

    def calc_value(self, sample):
        pointer = sample[0]
        value = self.aggregator(sample[pointer + 1: pointer + 1 + self.complexity + 1])
        return value

    def __getitem__(self, idx):
        sample = self.data[idx]
        value = self.calc_value(sample)

        return sample.long(), value.long()

    def __len__(self):
        return self.size

    def print(self):
        for idx, (sample, label) in enumerate(self):
            print(f'{idx}:\t {sample.tolist()} -> {label.tolist()}')


def main():
    p = 3
    ds = VectorPVR(10 ** p, complexity=2, holdout=2, aggregation_method='mod_sum', adversarial=False)
    print(p, ':', _get_tensor_size(ds.data))

    for idx in range(len(ds)):
        print(ds[idx])


if __name__ == '__main__':
    main()
