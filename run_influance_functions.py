from argparse import ArgumentParser

from xai_research.influance_function import compute_influence

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-r", "--run", type=str, default='m21q11cw',
                        help="run identifier in WandB")

    parser.add_argument("-e", "--epoch", type=int, default=75,
                        help="epoch checkpoint")

    parser.add_argument("-c", "--cuda", action='store_true',
                        help="run on GPU or not")

    parser.add_argument("-n", type=int, default=1, help="number of test points to run on.")

    args = parser.parse_args()

    compute_influence(run_id=args.run, epoch=args.epoch, num_test_points=args.n, use_cuda=args.cuda)
