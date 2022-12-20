from argparse import ArgumentParser
from pathlib import Path
import json
import cv2

import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

from tcav.model import ModelWrapper
from tcav.cav import get_avg_cav
import tcav.tcav as tcav
from tcav.utils import dot_product
from run_tcav import make_dirs, get_class_names, get_act_gen


def main(args):
    concepts = [v.strip() for v in args.concepts.split(",")]
    # this is a regularizer penalty parameter for linear classifier to get CAVs.
    alphas = [0.1]
    dirs = make_dirs(args)
    img_out_dir = dirs["results"] / f"visualisations/{args.target}"
    for concept in concepts:
        d = img_out_dir / concept
        d.mkdir(exist_ok=True, parents=True)

    bottlenecks = [bn.strip() for bn in args.layers.split(",")]
    bottlenecks = {bn: bn for bn in bottlenecks}

    model = create_model()
    class_names = get_class_names(dirs["source"])
    mymodel = ModelWrapper(model, bottlenecks, class_names)

    act_generator = get_act_gen(args, dirs, mymodel, concepts)
    data_path = Path(act_generator.prefix)

    mytcav = tcav.TCAV(
        args.target,
        concepts,
        bottlenecks,
        act_generator,
        alphas,
        cav_dir=dirs["cav"],
        num_random_exp=args.num_rand,
        do_random_pairs=False,
        grad_dir=dirs["grad"],
    )

    imgs, all_dot_acts, all_directional_derivatives = plot_all_visualisations(
        args.target, mytcav, act_generator, concepts, mymodel, data_path, img_out_dir
    )

    bottlenecks = list(mytcav.bottlenecks.values())
    for values, name in zip(
        [all_dot_acts, all_directional_derivatives],
        ["act_dot", "directional_derivative"],
    ):
        for c, concept in enumerate(concepts):
            plot_visualisation_array(
                imgs, values, concept, c, name, bottlenecks, img_out_dir
            )

    print("Done!")


def plot_visualisation_array(
    imgs, values, concept, concept_idx, name, bottlenecks, img_out_dir
):
    fig, axes = plt.subplots(
        len(bottlenecks),
        len(imgs),
        figsize=(2 * len(imgs), 2 * len(bottlenecks)),
    )
    for i, row in enumerate(axes):
        bn = bottlenecks[i]
        all_vs = [values[j][bn][concept_idx] for j in range(len(row))]
        max_v = max([np.abs(v).max() for v in all_vs])
        for j, ax in enumerate(row):
            ax.imshow(imgs[j])
            act_rescaled = rescale_array(values[j][bn][concept_idx], imgs[j])
            s = ax.imshow(
                act_rescaled, alpha=0.4, cmap="seismic", vmin=-max_v, vmax=max_v
            )
            ax.axis("off")
            if j == len(row) - 1:
                fig.colorbar(s, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(img_out_dir / concept / f"{name}_array.png")


def plot_all_visualisations(
    target, mytcav, act_generator, concepts, mymodel, data_path, img_out_dir
):
    examples = act_generator.get_examples_for_concept(target)
    img_paths = [v["path"] for v in act_generator.concept_dict[target].values()]
    params = {}
    for bn in mytcav.bottlenecks.values():
        params[bn] = [v for v in mytcav.params if v.bottleneck == bn]

    target_id = mymodel.label_to_id(target)
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
                act_dot, directional_derivatives = plot_avg_cav_visualisation(
                    bn_params,
                    concept,
                    img,
                    act,
                    grad,
                    img_out_dir,
                    img_path,
                    bn,
                )
                bn_acts.append(act_dot)
                bn_grads.append(directional_derivatives)
            img_grads[bn] = bn_grads
            img_acts[bn] = bn_acts
        all_directional_derivatives.append(img_grads)
        all_dot_acts.append(img_acts)
    return imgs, all_dot_acts, all_directional_derivatives


def get_avg_cav_responses(params, act, grad):
    avg_cav = get_avg_cav(params)

    directional_derivatives = dot_product(grad, avg_cav)
    act_dot = dot_product(act, avg_cav)
    return act_dot, directional_derivatives


def plot_avg_cav_visualisation(
    bn_params, concept, img, act, grad, img_out_dir, img_path, bn
):
    concept_params = [param for param in bn_params if concept in param.concepts]

    act_dot, directional_derivatives = get_avg_cav_responses(concept_params, act, grad)

    fig, ax = rescale_and_plot(directional_derivatives, img)
    plt.savefig(
        img_out_dir / concept / f"{img_path.stem}_{bn}_directional_derivative.png"
    )
    plt.close(fig)

    fig, ax = rescale_and_plot(act_dot, img)
    plt.savefig(img_out_dir / concept / f"{img_path.stem}_{bn}_act_dot.png")
    plt.close(fig)
    return act_dot, directional_derivatives


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


def cosine_sim(grad, cav):
    v = dot_product(grad, cav)
    norms = np.linalg.norm(grad[0], axis=0)
    cav_norm = np.linalg.norm(cav)
    v = v / (norms * cav_norm)
    return v


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
