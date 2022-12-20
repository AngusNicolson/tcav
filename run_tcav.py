from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

from tcav.model import ModelWrapper
from tcav.dataset import JsonDataset
import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.tcav as tcav
import tcav.utils as utils
import tcav.utils_plot as utils_plot  # utils_plot requires matplotlib


def main(args):
    exp_name = args.exp_name
    num_random_exp = args.num_rand
    target = args.target
    concepts = [v.strip() for v in args.concepts.split(",")]
    # this is a regularizer penalty parameter for linear classifier to get CAVs.
    alphas = [0.1]
    max_examples = args.max_examples
    num_workers = args.num_workers
    act_func = args.act_func
    overwrite = args.overwrite

    working_dir = Path(args.working_dir) / act_func
    activation_dir = working_dir / "activations"
    cav_dir = working_dir / "cavs"
    grad_dir = working_dir / "grads"

    source_dir = Path(args.source_dir)  # '/home/lina3782/labs/explain/imagenet'
    bottlenecks = [bn.strip() for bn in args.layers.split(",")]
    bottlenecks = {bn: bn for bn in bottlenecks}

    results_dir = source_dir / f"results/{exp_name}"

    for d in [activation_dir, working_dir, cav_dir, grad_dir, results_dir]:
        d.mkdir(exist_ok=True, parents=True)

    model = create_model()
    label_path = source_dir / "class_names.txt"
    with open(label_path, "r") as fp:
        class_names = fp.read()
    class_names = class_names.split("\n")
    class_names_short = [v.split(",")[0] for v in class_names]
    mymodel = ModelWrapper(model, bottlenecks, class_names_short)

    data_path = source_dir / "data"
    source_json = create_source_json(target, concepts, num_random_exp, data_path)
    source_json_path = results_dir / "source.json"
    with open(source_json_path, "w") as fp:
        json.dump(source_json, fp, indent=2)

    prefix = str(data_path) + "/"
    act_generator = act_gen.ActivationGenerator(
        mymodel,
        source_json_path,
        activation_dir,
        JsonDataset,
        max_examples=max_examples,
        prefix=prefix,
        num_workers=num_workers,
        act_func=act_func,
    )

    mytcav = tcav.TCAV(
        target,
        concepts,
        bottlenecks,
        act_generator,
        alphas,
        cav_dir=cav_dir,
        num_random_exp=num_random_exp,
        do_random_pairs=True,
        grad_dir=grad_dir,
    )

    print("Training CAVs...")
    print("This may take a while... Go get coffee!")
    mytcav.train_cavs(overwrite=overwrite)
    print("Training complete!")

    print("Running TCAV...")
    results = mytcav.run(overwrite=overwrite)

    fig = utils_plot.plot_results(
        results, num_random_exp=num_random_exp, figsize=(10, 5), show=False
    )
    fig.axes[0].axhline(0.5, color="gray", alpha=0.8, linestyle="--")
    plt.savefig(results_dir / f"{target}_tcav_scores.png")

    with open(results_dir / f"{target}.json", "w") as fp:
        json.dump(results, fp, indent=2)

    all_acc_means = {}
    all_acc_stds = {}
    for concept in concepts:
        accs = {}
        for bn in bottlenecks.keys():
            accs[bn] = [
                v["cav_accuracies"]["overall"]
                for v in results
                if (v["bottleneck"] == bn) and (v["cav_concept"] == concept)
            ]
        accs = np.array(list(accs.values()))
        accs_mean = accs.mean(axis=1)
        accs_std = accs.std(axis=1)

        all_acc_means[concept] = accs_mean
        all_acc_stds[concept] = accs_std

    fig, ax = plt.subplots()
    for concept in concepts:
        ax.plot(list(bottlenecks.keys()), all_acc_means[concept], label=concept)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=45)
    plt.legend(frameon=False)
    plt.savefig(results_dir / f"cav_accuracies.png")

    print("Done!")


def create_model(dataset="imagenet"):
    if dataset == "imagenet":
        return models.resnet50(pretrained=True)


def create_source_json(target, concepts, num_random_exp, data_path):
    source_json = {}
    for concept in (
        [target] + concepts + [f"random500_{i}" for i in range(num_random_exp)]
    ):
        paths = (data_path / concept).glob("*.jpg")
        source_json[concept] = {
            i: {"path": f"{concept}/{path.name}", "label": 0}
            for i, path in enumerate(paths)
        }
    return source_json


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--working-dir",
        help="Where to save intermediate outputs",
        default="/tmp/tcav/example",
    )
    parser.add_argument("--source-dir", help="The location of the images")
    parser.add_argument(
        "--layers",
        help="A comma seperated list of the layers to run TCAV on",
        default="layer4.0, layer4.1",
    )
    parser.add_argument(
        "--num-rand",
        help="The number of random experiments to run",
        type=int,
        default=30,
    )
    parser.add_argument("--target", help="The target class name")
    parser.add_argument(
        "--concepts",
        help="A comma seperated list of the concepts you wish to create CAVs for",
    )
    parser.add_argument(
        "--max-examples",
        help="The maximum number of images per concept",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--num-workers",
        help="The number of cpu workers for dataloading",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--exp-name", help="Experiment name (for saving results)", default="example"
    )
    parser.add_argument(
        "--act-func",
        help="Optional function to apply to the model activations",
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        help="Overwrite activations, CAVs and gradients",
        action="store_true",
    )
    main(parser.parse_args())
