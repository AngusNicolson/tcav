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
    concepts = [v.strip() for v in args.concepts.split(",")]
    # this is a regularizer penalty parameter for linear classifier to get CAVs.
    alphas = [0.1]
    dirs = make_dirs(args)
    bottlenecks = [bn.strip() for bn in args.layers.split(",")]
    bottlenecks = {bn: bn for bn in bottlenecks}

    model = create_model()
    class_names = get_class_names(dirs["source"])
    mymodel = ModelWrapper(model, bottlenecks, class_names)

    act_generator = get_act_gen(args, dirs, mymodel, concepts)

    mytcav = tcav.TCAV(
        args.target,
        concepts,
        bottlenecks,
        act_generator,
        alphas,
        cav_dir=dirs["cav"],
        num_random_exp=args.num_rand,
        do_random_pairs=True,
        grad_dir=dirs["grad"],
    )

    print("Training CAVs...")
    print("This may take a while... Go get coffee!")
    mytcav.train_cavs(overwrite=args.overwrite)
    print("Training complete!")

    print("Running TCAV...")
    results = mytcav.run(overwrite=args.overwrite)

    fig = utils_plot.plot_results(
        results, num_random_exp=args.num_rand, figsize=(10, 5), show=False
    )
    plt.savefig(dirs["results"] / f"{args.target}_tcav_scores.png")

    with open(dirs["results"] / f"{args.target}.json", "w") as fp:
        json.dump(results, fp, indent=2)

    acc_means, acc_stds = utils.get_cav_accuracies_from_results(
        results, concepts, bottlenecks
    )

    fig, ax = utils_plot.plot_cav_accuracies(acc_means, concepts, bottlenecks)
    plt.savefig(dirs["results"] / f"cav_accuracies.png")

    print("Done!")


def get_act_gen(args, dirs, mymodel, concepts):
    data_path = dirs["source"] / "data"
    source_json = create_source_json(args.target, concepts, args.num_rand, data_path)
    source_json_path = dirs["results"] / "source.json"
    with open(source_json_path, "w") as fp:
        json.dump(source_json, fp, indent=2)

    prefix = str(data_path) + "/"
    act_generator = act_gen.ActivationGenerator(
        mymodel,
        source_json_path,
        dirs["activation"],
        JsonDataset,
        max_examples=args.max_examples,
        prefix=prefix,
        num_workers=args.num_workers,
        act_func=args.act_func,
    )
    return act_generator


def get_class_names(source_dir):
    label_path = source_dir / "class_names.txt"
    with open(label_path, "r") as fp:
        class_names = fp.read()
    class_names = class_names.split("\n")
    # Use shortened version for ImageNet because sometimes the list is very long
    class_names_short = [v.split(",")[0] for v in class_names]
    return class_names_short


def make_dirs(args):
    working_dir = Path(args.working_dir) / args.exp_name
    activation_dir = working_dir / "activations"
    cav_dir = working_dir / "cavs"
    grad_dir = working_dir / "grads"
    source_dir = Path(args.source_dir)
    results_dir = source_dir / f"results/{args.exp_name}"

    dirs = {
        "activation": activation_dir,
        "working": working_dir,
        "cav": cav_dir,
        "grad": grad_dir,
        "results": results_dir,
    }

    for d in dirs.values():
        d.mkdir(exist_ok=True, parents=True)
    return dirs


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
