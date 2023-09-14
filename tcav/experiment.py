
import json
import yaml

import torch
from torch.utils.data import DataLoader
from torchvision import models
import torchvision.transforms.transforms
from ldm.util import instantiate_from_config

from tcav.model import ModelWrapper
from tcav.utils import make_dirs
from tcav.dataset import JsonDataset
import tcav.activation_generator as act_gen
from tcav.tcav import TCAV


def setup_experiment(args):
    concepts = [v.strip() for v in args.concepts.split(",")]
    # this is a regularizer penalty parameter for linear classifier to get CAVs.
    alphas = [0.1]
    dirs = make_dirs(args)
    bottlenecks = [bn.strip() for bn in args.layers.split(",")]
    bottlenecks = {bn: bn for bn in bottlenecks}

    if args.model_path is not None:
        model = create_model(path=args.model_path, ldm_config_path=args.ldm_config)
    else:
        model = create_model("imagenet")
    class_names = get_class_names(dirs["source"])
    mymodel = ModelWrapper(model, bottlenecks, class_names)

    act_generator = get_act_gen(args, dirs, mymodel, concepts)

    mytcav = TCAV(
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

    return mytcav, dirs


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


def load_model_config(model_path, config_path, remove_loss=True):
    with open(config_path, "r") as fp:
        config = yaml.safe_load(fp)
    config = config["model"]
    if remove_loss:
        config["params"]["lossconfig"] = {'target': 'torch.nn.Identity'}
    config["params"]["ckpt_path"] = model_path
    config.pop("base_learning_rate")
    return config


def get_class_names(source_dir):
    label_path = source_dir / "class_names.txt"
    with open(label_path, "r") as fp:
        class_names = fp.read()
    class_names = class_names.split("\n")
    # Use shortened version for ImageNet because sometimes the list is very long
    class_names_short = [v.split(",")[0] for v in class_names]
    return class_names_short


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

