
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt

import tcav.utils as utils
import tcav.utils_plot as utils_plot  # utils_plot requires matplotlib
from tcav.experiment import setup_experiment, load_examples

from train_model import MyModel


def main(args):
    mytcav, dirs = setup_experiment(args)

    if not args.no_training:
        print("Training CAVs...")
        print("This may take a while... Go get coffee!")
        mytcav.train_cavs(overwrite=args.overwrite)
        print("Training complete!")

    print("Running TCAV...")
    if args.alternate_target_examples is None:
        results = mytcav.run(overwrite=args.overwrite)
    else:
        examples = load_examples(
            Path(args.alternate_target_examples), mytcav.act_generator, args.max_examples
        )
        results = mytcav.run_on_examples(
            examples, overwrite=args.overwrite, grad_suffix="_alternate"
        )

    fig = utils_plot.plot_results(
        results, num_random_exp=args.num_rand, figsize=(10, 5), show=False
    )
    plt.savefig(dirs["results"] / f"{args.target}{args.suffix}_tcav_scores.png")

    with open(dirs["results"] / f"{args.target}{args.suffix}.json", "w") as fp:
        json.dump(results, fp, indent=2)

    acc_means, acc_stds = utils.get_cav_accuracies_from_results(
        results, mytcav.concepts, mytcav.bottlenecks
    )

    fig, ax = utils_plot.plot_cav_accuracies(acc_means, mytcav.concepts, mytcav.bottlenecks)
    plt.tight_layout()
    plt.savefig(dirs["results"] / f"cav_accuracies.png")

    print("Done!")


if __name__ == "__main__":
    parser = utils.get_parser()
    parser.add_argument(
        "--overwrite",
        help="Overwrite activations, CAVs and gradients",
        action="store_true",
    )
    parser.add_argument(
        "--alternate-target-examples",
        help="Optional path to a directory containing target images",
        default=None,
    )
    parser.add_argument(
        "--no-training", help="Don't train any new CAVs", action="store_true"
    )
    main(parser.parse_args())
