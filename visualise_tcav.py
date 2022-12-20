from argparse import ArgumentParser
from pathlib import Path
import json
import cv2

import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

from tcav.model import ModelWrapper
from tcav.dataset import JsonDataset
import tcav.activation_generator as act_gen
from tcav.cav import CAV
import tcav.tcav as tcav


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

    working_dir = Path(args.working_dir) / act_func
    activation_dir = working_dir / "activations"
    cav_dir = working_dir / "cavs"
    grad_dir = working_dir / "grads"

    source_dir = Path(args.source_dir)  # '/home/lina3782/labs/explain/imagenet'
    bottlenecks = [bn.strip() for bn in args.layers.split(",")]
    bottlenecks = {bn: bn for bn in bottlenecks}

    results_dir = source_dir / f"results/{exp_name}"
    img_out_dir = results_dir / f"visualisations/{target}"
    for concept in concepts:
        d = img_out_dir / concept
        d.mkdir(exist_ok=True, parents=True)

    for d in [activation_dir, working_dir, cav_dir, grad_dir, results_dir, img_out_dir]:
        d.mkdir(exist_ok=True, parents=True)

    model = create_model()
    label_path = source_dir / "class_names.txt"
    with open(label_path, "r") as fp:
        class_names = fp.read()
    class_names = class_names.split("\n")
    class_names_short = [v.split(",")[0] for v in class_names]
    mymodel = ModelWrapper(model, bottlenecks, class_names_short)

    data_path = source_dir / "data"

    source_json_path = results_dir / "source.json"
    with open(source_json_path, "r") as fp:
        source_json = json.load(fp)

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
        do_random_pairs=False,
        grad_dir=grad_dir,
    )

    target_id = mymodel.label_to_id(target)
    examples = act_generator.get_examples_for_concept(target)
    img_paths = [v["path"] for v in source_json[target].values()]
    params = {}
    for bn in bottlenecks:
        params[bn] = [v for v in mytcav.params if v.bottleneck == bn]

    imgs = []
    all_directional_derivatives = []
    all_dot_acts = []
    for i, example in enumerate(examples):
        example = example.to("cuda")
        img_path = data_path / img_paths[i]
        img = load_img(img_path)
        imgs.append(img)
        img_acts = {}
        img_grads = {}
        for bn, bn_params in params.items():
            bn_acts = []
            bn_grads = []
            grad = mymodel.get_gradient(example, target_id, bn).cpu().numpy()
            act = mymodel.bottlenecks_tensors[bn].cpu().detach().numpy()
            for concept in concepts:
                cav = get_cav(bn_params[0])
                avg_cav = np.zeros(cav.shape)

                j = 0
                for param in bn_params:
                    if concept in param.concepts:
                        j += 1
                        cav = get_cav(param)
                        avg_cav += cav

                avg_cav = avg_cav / j
                directional_derivatives = dot_product(grad, avg_cav)
                act_dot = dot_product(act, avg_cav)
                bn_acts.append(act_dot)
                bn_grads.append(directional_derivatives)

                fig, ax = rescale_and_plot(directional_derivatives, img)
                plt.savefig(
                    img_out_dir
                    / concept
                    / f"{img_path.stem}_{bn}_directional_derivative.png"
                )
                plt.close(fig)

                fig, ax = rescale_and_plot(act_dot, img)
                plt.savefig(img_out_dir / concept / f"{img_path.stem}_{bn}_act_dot.png")
                plt.close(fig)
            img_grads[bn] = bn_grads
            img_acts[bn] = bn_acts
        all_directional_derivatives.append(img_grads)
        all_dot_acts.append(img_acts)

    bottlenecks = list(bottlenecks.values())
    for values, name in zip(
        [all_dot_acts, all_directional_derivatives],
        ["act_dot", "directional_derivative"],
    ):
        for c, concept in enumerate(concepts):
            fig, axes = plt.subplots(
                len(bottlenecks),
                len(imgs),
                figsize=(2 * len(imgs), 2 * len(bottlenecks)),
            )
            for i, row in enumerate(axes):
                bn = bottlenecks[i]
                all_vs = [values[j][bn][c] for j in range(len(row))]
                max_v = max([np.abs(v).max() for v in all_vs])
                for j, ax in enumerate(row):
                    ax.imshow(imgs[j])
                    act_rescaled = rescale_array(values[j][bn][c], imgs[j])
                    s = ax.imshow(
                        act_rescaled, alpha=0.4, cmap="seismic", vmin=-max_v, vmax=max_v
                    )
                    ax.axis("off")
                    if j == len(row) - 1:
                        fig.colorbar(s, ax=ax, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.savefig(img_out_dir / concept / f"{name}_array.png")

    print("Done!")


def load_img(path):
    img = cv2.imread(str(path))
    img = img[..., ::-1]
    return img


def rescale_and_plot(v, img):
    act_rescaled = rescale_array(v, img)
    fig, ax = plt.subplots()
    ax.imshow(img)
    s = ax.imshow(act_rescaled, alpha=0.4, cmap="seismic")
    fig.colorbar(s, ax=ax)
    ax.axis("off")
    return fig, ax


def rescale_array(v, img):
    return cv2.resize(v, dsize=img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)


def dot_product(v, cav):
    """Calculate the dot product for each 1x1xD block of activations
    v (np.Array): Gradient or activations of a single sample (H,W,D)
    cav (np.Array): Direction of the CAV (D,)
    """
    return np.dot(v[0].T, cav).T


def cosine_sim(grad, cav):
    v = dot_product(grad, cav)
    norms = np.linalg.norm(grad[0], axis=0)
    cav_norm = np.linalg.norm(cav)
    v = v / (norms * cav_norm)
    return v


def get_cav(param):
    bottleneck = param.bottleneck
    concepts = param.concepts
    alpha = param.alpha
    cav_dir = param.cav_dir

    # Get CAVs
    cav_hparams = CAV.default_hparams()
    cav_hparams["alpha"] = alpha
    a_cav_key = CAV.cav_key(
        concepts, bottleneck, cav_hparams["model_type"], cav_hparams["alpha"]
    )

    cav_path = cav_dir / (a_cav_key.replace("/", ".") + ".pkl")
    cav_instance = CAV.load_cav(cav_path)
    cav_concept = concepts[0]
    direction = cav_instance.get_direction(cav_concept)
    return direction


def create_model():
    return models.resnet50(pretrained=True)


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
        default=5,
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
    main(parser.parse_args())
