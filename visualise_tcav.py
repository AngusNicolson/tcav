from argparse import ArgumentParser
from pathlib import Path

from torchvision import models

from tcav.model import ModelWrapper
import tcav.tcav as tcav

from tcav.visualise import (
    get_all_responses,
    plot_all_visualisations,
    plot_visualisation_array,
)
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

    img_paths = [v["path"] for v in act_generator.concept_dict[args.target].values()]
    imgs, all_dot_acts, all_directional_derivatives = get_all_responses(
        args.target, img_paths, mytcav, act_generator, concepts, mymodel, data_path
    )

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
                imgs, values, concept, c, name, bottlenecks, img_out_dir
            )

    print("Done!")


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
