from pathlib import Path

import numpy as np
from torchvision import models

from tcav.model import ModelWrapper
import tcav.tcav as tcav
import tcav.utils as utils

from tcav.visualise import (
    get_all_responses,
    plot_all_visualisations,
    plot_visualisation_array,
)
from run_tcav import get_class_names, get_act_gen, create_model
from train_model import MyModel


def main(args):
    concepts = [v.strip() for v in args.concepts.split(",")]
    # this is a regularizer penalty parameter for linear classifier to get CAVs.
    alphas = [0.1]
    dirs = utils.make_dirs(args)
    img_out_dir = dirs["results"] / f"visualisations/{args.target}"
    for concept in concepts:
        d = img_out_dir / concept
        d.mkdir(exist_ok=True, parents=True)

    bottlenecks = [bn.strip() for bn in args.layers.split(",")]
    bottlenecks = {bn: bn for bn in bottlenecks}

    model = create_model(path=args.model_path, ldm_config_path=args.ldm_config)
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

    if args.dataset_config is None:
        img_paths = [
            Path(v["path"]) for v in act_generator.concept_dict[args.target].values()
        ]
    else:
        img_paths = None
    imgs, all_dot_acts, all_directional_derivatives = get_all_responses(
        args.target, img_paths, mytcav, act_generator, concepts, mymodel, data_path
    )

    dot_acts_to_save = rearrange_acts(all_dot_acts, concepts)
    for c in concepts:
        for bn, bn_acts in dot_acts_to_save[c].items():
            np.save(img_out_dir / c / f"{bn}_dot_acts.npy", bn_acts)

    if not args.no_plots:
        plot_all_visualisations(
            imgs,
            all_dot_acts,
            all_directional_derivatives,
            bottlenecks,
            concepts,
            img_paths,
            img_out_dir,
        )

        bottlenecks = list(mytcav.bottlenecks.values())
        for values, name in zip(
            [all_dot_acts, all_directional_derivatives],
            ["act_dot", "directional_derivative"],
        ):
            for c, concept in enumerate(concepts):
                plot_visualisation_array(
                    imgs, values, c, f"{concept}/{name}", bottlenecks, img_out_dir
                )

    print("Done!")


def rearrange_acts(acts, concepts):
    out = {c: {} for c in concepts}
    for img_acts in acts:
        for i, c in enumerate(concepts):
            for bn, bn_acts in img_acts.items():
                if bn not in out[c].keys():
                    out[c][bn] = [bn_acts[i][np.newaxis, :, :]]
                else:
                    out[c][bn].append(bn_acts[i][np.newaxis, :, :])
    for c in concepts:
        for bn, bn_acts in out[c].items():
            out[c][bn] = np.concatenate(bn_acts)
    return out


if __name__ == "__main__":
    parser = utils.get_parser()
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Don't save any plots",
    )
    main(parser.parse_args())
