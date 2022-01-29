from math import factorial

from datasets import VectorPVR
import numpy as np

# parameters
SIZE = 10 ** 3
COMPLEXITY = 1
HOLDOUT = 2
AGGREGATION_METHOD = 'mod_sum'


def test_non_adversarial():
    _test_vectorized(SIZE, COMPLEXITY, HOLDOUT, AGGREGATION_METHOD, False)


def test_adversarial():
    _test_vectorized(SIZE, COMPLEXITY, HOLDOUT, AGGREGATION_METHOD, True)


def _test_vectorized(size: int, complexity: int, holdout: int, aggregation_method: str, adversarial: bool):
    ds = VectorPVR(size,
                   complexity=complexity,
                   holdout=holdout,
                   aggregation_method=aggregation_method,
                   adversarial=adversarial
                   )

    holdout_permutations = ds.get_holdout_permutations()

    counters = np.zeros(10, dtype=np.uint)

    for idx, (sample, label) in enumerate(ds):
        # validate holdout
        if holdout > 0:
            pointer = sample[0]
            value_window = tuple(sample[pointer + 1: pointer + 1 + complexity + 1].tolist())
            if adversarial:
                # check that only the permutations are in the window
                assert value_window in holdout_permutations
            else:
                # check that non of the permutations appear in the window
                assert value_window not in holdout_permutations

        # calculate stats
        for x in sample[1:]:
            # must loop to take into account multiple appearances in single sample
            counters[x] += 1

    counters_percentage = counters / np.sum(counters) * 100

    print('\n')
    print("Digit appearance percentage:")
    print(counters_percentage)

