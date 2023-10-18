from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tcav.cav import get_avg_cav
from tcav.utils import dot_product


def get_all_responses(
    target, img_paths, mytcav, act_generator, concepts, mymodel, data_path
):
    if img_paths is None:
        load_imgs_from_disk = False
    else:
        load_imgs_from_disk = True
    examples = act_generator.get_examples_for_concept(target)
    params = {}
    for bn in mytcav.bottlenecks:
        params[bn] = [v for v in mytcav.params if v.bottleneck == bn]

    target_id = mymodel.label_to_id(target)
    imgs = []
    all_directional_derivatives = []
    all_dot_acts = []
    for i, example in enumerate(examples):
        example = example.to("cuda")
        if load_imgs_from_disk:
            img_path = data_path / img_paths[i]
            img = load_img(img_path)
        else:
            img = example.detach().cpu().numpy().transpose(1, 2, 0)
        imgs.append(img)
        img_acts = {}
        img_grads = {}
        for bn, bn_params in params.items():
            bn_acts = []
            bn_grads = []
            grad = mymodel.get_gradient(example, target_id, bn).cpu().numpy()
            act = mymodel.bottlenecks_tensors[bn].cpu().detach().numpy()
            for concept in concepts:
                act_dot, directional_derivatives = get_avg_cav_responses_for_concept(
                    concept, bn_params, act, grad
                )
                bn_acts.append(act_dot)
                bn_grads.append(directional_derivatives)
            img_grads[bn] = bn_grads
            img_acts[bn] = bn_acts
        all_directional_derivatives.append(img_grads)
        all_dot_acts.append(img_acts)
    return imgs, all_dot_acts, all_directional_derivatives


def plot_all_visualisations(
    imgs,
    all_dot_acts,
    all_directional_derivatives,
    bottlenecks,
    concepts,
    img_paths,
    img_out_dir,
):
    if img_paths is None:
        img_paths = [Path(f"{i:03d}.png")for i in range(len(imgs))]
    for i, img in enumerate(imgs):
        for bn in bottlenecks:
            for c, concept in enumerate(concepts):
                plot_avg_cav_visualisation(
                    img,
                    all_dot_acts[i][bn][c],
                    all_directional_derivatives[i][bn][c],
                    img_out_dir / concept,
                    f"{img_paths[i].stem}_{bn}",
                )


def get_avg_cav_responses(params, act, grad):
    avg_cav = get_avg_cav(params)

    directional_derivatives = dot_product(grad, avg_cav)
    act_dot = dot_product(act, avg_cav)
    return act_dot, directional_derivatives


def get_avg_cav_responses_for_concept(concept, params, act, grad):
    concept_params = [param for param in params if concept in param.concepts]
    act_dot, directional_derivatives = get_avg_cav_responses(concept_params, act, grad)
    return act_dot, directional_derivatives


def plot_avg_cav_visualisation(
    img, act_dot, directional_derivatives, img_out_dir, name
):
    fig, ax = rescale_and_plot(directional_derivatives, img)
    plt.savefig(img_out_dir / f"{name}_directional_derivative.png")
    plt.close(fig)

    fig, ax = rescale_and_plot(act_dot, img)
    plt.savefig(img_out_dir / f"{name}_act_dot.png")
    plt.close(fig)


def plot_visualisation_array(imgs, values, concept_idx, name, bottlenecks, img_out_dir):
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
    plt.savefig(img_out_dir / f"{name}_array.png")


def load_img(path):
    img = cv2.imread(str(path))
    img = img[..., ::-1]
    return img


def rescale_and_plot(v, img, scale_min_max=True, cmap="seismic"):
    act_rescaled = rescale_array(v, img)
    fig, ax = plt.subplots()
    ax.imshow(img)
    max_v = np.abs(act_rescaled).max()
    if scale_min_max:
        vmin, vmax = -max_v, max_v
    else:
        vmin, vmax = None, None
    s = ax.imshow(act_rescaled, alpha=0.4, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(s, ax=ax)
    ax.axis("off")
    return fig, ax


def rescale_array(v, img):
    return cv2.resize(v, dsize=img.shape[:2][::-1], interpolation=cv2.INTER_CUBIC)
