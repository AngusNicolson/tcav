from argparse import ArgumentParser
from pathlib import Path
import json

import torchvision.transforms.transforms
import yaml

import numpy as np
import matplotlib.pyplot as plt
from torchvision import models
from torch.utils.data import DataLoader
import torch

from tcav.model import ModelWrapper
from tcav.dataset import JsonDataset
import tcav.activation_generator as act_gen
import tcav.cav as cav
import tcav.tcav as tcav
import tcav.utils as utils
import tcav.utils_plot as utils_plot  # utils_plot requires matplotlib

from train_model import MyModel
from ldm.util import instantiate_from_config


def main(args):
    concepts = [v.strip() for v in args.concepts.split(",")]
    # this is a regularizer penalty parameter for linear classifier to get CAVs.
    alphas = [0.1]
    dirs = utils.make_dirs(args)
    bottlenecks = [bn.strip() for bn in args.layers.split(",")]
    bottlenecks = {bn: bn for bn in bottlenecks}

    if args.model_path is not None:
        model = create_model(path=args.model_path, ldm_config_path=args.ldm_config)
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
        perturb=args.perturb,
    )

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
            Path(args.alternate_target_examples), act_generator, args.max_examples
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
        results, concepts, bottlenecks
    )

    fig, ax = utils_plot.plot_cav_accuracies(acc_means, concepts, bottlenecks)
    plt.tight_layout()
    plt.savefig(dirs["results"] / f"cav_accuracies.png")

    print("Done!")


def load_model_config(model_path, config_path, remove_loss=True):
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    config = config["model"]
    if remove_loss:
        config["params"]["lossconfig"] = {'target': 'torch.nn.Identity'}
    config["params"]["ckpt_path"] = model_path
    config.pop("base_learning_rate")
    return config


def load_examples(examples_dir, act_generator, n):
    paths = examples_dir.glob("*.jpg")
    source_json = {
        "examples": {
            i: {"path": f"/{path.name}", "label": 0} for i, path in enumerate(paths)
        }
    }
    dataset = act_generator.dataset_class(source_json, "examples", str(examples_dir))
    dataloader = DataLoader(
        dataset, n, shuffle=False, num_workers=act_generator.num_workers
    )
    for sample in dataloader:
        imgs = sample[0]
        break
    return imgs


def get_act_gen(args, dirs, mymodel, concepts):
    data_path = dirs["source"] / "data"
    source_json = create_source_json(args.target, concepts, args.num_rand, data_path)
    source_json_path = dirs["results"] / "source.json"
    with open(source_json_path, "w") as fp:
        json.dump(source_json, fp, indent=2)

    if args.dataset_config is not None:
        with open(args.dataset_config, "r") as fp:
            dataset_config = yaml.safe_load(fp)
        dataset_creator = instantiate_from_config(dataset_config["dataset"])
        dataset_class = lambda json_file, split, prefix: dataset_creator(split)
        data_saved_on_disk = False
    elif args.no_normalize:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.transforms.ToTensor(),
            torchvision.transforms.Resize((256, 256)),
        ])
        dataset_class = lambda json_file, split, prefix: JsonDataset(
            json_file, split, prefix, transform=transform
        )
        data_saved_on_disk = True
    else:
        dataset_class = JsonDataset
        data_saved_on_disk = True

    prefix = str(data_path) + "/"
    act_generator = act_gen.ActivationGenerator(
        mymodel,
        source_json_path,
        dirs["activation"],
        dataset_class=dataset_class,
        max_examples=args.max_examples,
        prefix=prefix,
        num_workers=args.num_workers,
        act_func=args.act_func,
        data_saved_on_disk=data_saved_on_disk,
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


def create_model(dataset="imagenet", path=None, ldm_config_path=None):
    if ldm_config_path is not None:
        if path is None:
            raise ValueError(
                "Need a path for the model .ckpt as well as the model config!"
            )
        config = load_model_config(path, ldm_config_path)
        return instantiate_from_config(config)
    if path is not None:
        return torch.load(path)
    if dataset == "imagenet":
        return models.resnet50(pretrained=True)
    raise ValueError(
        "Model could not be loaded. Either no path or an unrecognised dataset."
    )


def create_source_json(target, concepts, num_random_exp, data_path):
    source_json = {}
    for concept in (
        [target] + concepts + [f"random500_{i}" for i in range(num_random_exp)]
    ):
        for ext in [".jpg", ".png"]:
            paths = (p for p in (data_path / concept).glob("**/*") if p.suffix in {".jpg", ".png"})
        source_json[concept] = {
            i: {"path": f"{concept}/{path.name}", "label": 0}
            for i, path in enumerate(paths)
        }
    return source_json


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
