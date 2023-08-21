from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from tcav.model import ModelWrapper
from tcav.utils import get_parser, make_dirs
import tcav.tcav as tcav
from tcav.cav import CAV

from torchinfo import summary

from run_tcav import get_class_names, get_act_gen, create_model, \
    load_examples


def main(args):
    if args.perturb is None:
        raise ValueError("--perturb must be set to some float value")
    perturb = args.perturb

    concepts = [v.strip() for v in args.concepts.split(",")]
    alphas = [0.1]
    dirs = make_dirs(args)
    img_out_dir = dirs["results"] / f"perturbed_generations/{args.target}"
    for concept in concepts:
        d = img_out_dir / concept
        d.mkdir(exist_ok=True, parents=True)

    bottlenecks = [bn.strip() for bn in args.layers.split(",")]
    bottlenecks = {bn: bn for bn in bottlenecks}

    # /home/lina3782/labs/explain/imagenet/models/vqgan/vqgan_imagenet_f16_1024/last.ckpt
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
        perturb=args.perturb
    )

    img_dir = Path("/home/lina3782/labs/explain/imagenet/data/horse2zebra/testA/")
    #img_dir = Path("/home/lina3782/labs/explain/element_dataset/simple_all/data_256/random_examples")
    examples = load_examples(img_dir, act_generator, 4)
    bn = "decoder.mid.block_2"  # "decoder.up.3.block.2" # "encoder.conv_out"
    #base_save_path = Path("/home/lina3782/labs/explain/element_dataset/simple_all/results/ae/maxpool/perturbed_generations/")
    base_save_path = Path("/home/lina3782/labs/explain/imagenet/results/vqgan/f16/maxpool/perturbed_generations/")

    fig, axes = plt.subplots(1, len(examples))
    for i, ax in enumerate(axes):
        ax.imshow(examples[i].permute(1, 2, 0))
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(base_save_path / "example_inputs.png")
    plt.show()

    with torch.no_grad():
        outputs = model(examples)[0].cpu()

    fig, axes = plt.subplots(2, len(examples))
    for i, ax in enumerate(axes[0]):
        ax.imshow(examples[i].permute(1, 2, 0))
        ax.axis("off")
    for i, ax in enumerate(axes[1]):
        ax.imshow(outputs[i].permute(1, 2, 0).cpu())
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(base_save_path / "example_outputs.png")
    plt.show()

    for concept in concepts:
        #cav_path = f"/mnt/data01/lina3782/element/tcav/simple_all/ae/maxpool/cavs/{concept}-random500_0-{bn}-linear-0.1.pkl"
        cav_path = f"/mnt/data01/lina3782/imagenet/tcav/vqgan/f16/maxpool/cavs/{concept}-random500_0-{bn}-linear-0.1.pkl"

        cav = CAV.load_cav(cav_path)

        activations = mytcav.activation_generator.get_activations_for_examples(
            examples, bn, grad=False, use_act_func=False
        )

        for perturb in [1, 5, 10, 15, 20, 25, 50]:
            cav_dir = cav.get_direction(cav.concepts[0]).astype("float32")
            cav_dir = cav_dir.reshape((cav_dir.shape[0], 1, 1))
            perturbations = activations + perturb * cav_dir

            with torch.no_grad():
                perturbed_outputs = decoder_forward_from_upsampling(mymodel.model.decoder, perturbations.to("cuda"))
            fig, axes = plt.subplots(2, len(examples))
            for i, ax in enumerate(axes[0]):
                ax.imshow(examples[i].permute(1, 2, 0))
                ax.axis("off")
            for i, ax in enumerate(axes[1]):
                ax.imshow(perturbed_outputs[i].permute(1, 2, 0).cpu())
                ax.axis("off")
            plt.tight_layout()
            plt.savefig(base_save_path / f"example_perturbed_change_{concept}_{perturb:02d}.png")

    with torch.no_grad():
        posterior = model.encode(examples)
        z = posterior.sample()
        dec = model.decode(z)

        z_post_quant = model.post_quant_conv(z)
        activations = decoder_forward_pre_upsampling(model.decoder, z_post_quant)

    concept = "stripes_diagonal"
    cav_path = f"/mnt/data01/lina3782/element/tcav/simple_all/ae/maxpool/cavs/{concept}-random500_0-{bn}-linear-0.1.pkl"
    cav = CAV.load_cav(cav_path)


    cav_dir = cav.get_direction(cav.concepts[0]).astype("float32")
    cav_dir = cav_dir.reshape((cav_dir.shape[0], 1, 1))
    perturb = 30
    activations = mytcav.activation_generator.get_activations_for_examples(
        examples, bn, grad=False, use_act_func=False
    )
    # (B, C, H [top:bottom], W[left:right])
    activations[:, :, 10:, 3:9] += perturb * cav_dir
    perturbations = activations
    with torch.no_grad():
        perturbed_outputs = decoder_forward_from_upsampling(
            mymodel.model.decoder,
            perturbations.to("cuda")
        )

    fig, axes = plt.subplots(2, len(examples))
    for i, ax in enumerate(axes[0]):
        #for j in range(16):
            #ax.axvline(j * (256 / 16), color="gray")
        ax.imshow(examples[i].permute(1, 2, 0))
        ax.axis("off")

    for i, ax in enumerate(axes[1]):
        #for j in range(16):
            #ax.axhline(j * (256 / 16), color="gray")
        ax.imshow(perturbed_outputs[i].permute(1, 2, 0).cpu())
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    concepts = concepts + ["red", "green", "blue"] + ['square', 'circle', 'triangle', 'plus']

    cavs = load_cavs(
        concepts,
        "/mnt/data01/lina3782/element/tcav/simple_all/ae/maxpool/cavs",
        bn
    )

    activations = mytcav.activation_generator.get_activations_for_examples(
        examples, bn, grad=False, use_act_func=False
    )
    # (B, C, H [top:bottom], W[left:right])
    # 10:, 3:9 red solid circle bottom left
    # 9:, :5 blue spotty circle bottom left second image
    change = - 15*cavs["blue"]
    activations[:, :, 9:, :5] += change
    #10 * cavs["stripes_diagonal"] - 10 * cavs["spots_polka"] + 5 * cavs["red"] - 5 * cavs["blue"]
    perturbations = activations
    with torch.no_grad():
        perturbed_outputs = decoder_forward_from_upsampling(
            mymodel.model.decoder,
            perturbations.to("cuda")
        )

    plot_single_comparison(
        examples[2].permute(1, 2, 0),
        perturbed_outputs[2].permute(1, 2, 0).cpu()
    )
    plt.savefig(base_save_path / "attempted_deletion_example_minus_blue_goes_green.png")
    plt.show()

    fig, axes = plt.subplots(2, len(examples))
    for i, ax in enumerate(axes[0]):
        #for j in range(16):
            #ax.axvline(j * (256 / 16), color="gray")
        ax.imshow(examples[i].permute(1, 2, 0))
        ax.axis("off")

    for i, ax in enumerate(axes[1]):
        #for j in range(16):
            #ax.axhline(j * (256 / 16), color="gray")
        ax.imshow(perturbed_outputs[i].permute(1, 2, 0).cpu())
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    examples = torch.ones((1, 3, 256, 256))

    for color in ["red", "green", "blue"]:
        activations = mytcav.activation_generator.get_activations_for_examples(
            examples, bn, grad=False, use_act_func=False
        )
        # (B, C, H [top:bottom], W[left:right])
        # 10:, 3:9 red solid circle bottom left
        # 9:, :5 blue spotty circle bottom left second image
        change = perturb * cavs["square"] + perturb / 2 * cavs[color] + perturb * cavs[
            "solid"]
        activations[:, :, 2:6, 2:6] += change
        activations[:, :, 2:6, 10:14] += change
        activations[:, :, 10:14, 2:6] += change
        activations[:, :, 10:14, 10:14] += change
        # 10 * cavs["stripes_diagonal"] - 10 * cavs["spots_polka"] + 5 * cavs["red"] - 5 * cavs["blue"]
        perturbations = activations
        with torch.no_grad():
            perturbed_outputs = decoder_forward_from_upsampling(
                mymodel.model.decoder,
                perturbations.to("cuda")
            )

        plot_single_comparison(
            examples[0].permute(1, 2, 0),
            perturbed_outputs[0].permute(1, 2, 0).cpu()
        )
        # plt.savefig(base_save_path / "blue_spots_to_red_stripes_single_object_change.png")
        plt.show()

    print("Done!")


def preprocess(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    assert x.size(1) == 3
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = (x[:, i, :, :] - mean[i]) / std[i]
    return y

def undo_preprocess(x):
    assert x.size(1) == 3
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    y = torch.zeros_like(x)
    for i in range(3):
        y[:, i, :, :] = x[:, i, :, :] * std[i] + mean[i]
    return y

def load_cavs(concepts, base_path, layer):
    cav_paths = {k: f"{base_path}/{k}-random500_0-{layer}-linear-0.1.pkl" for k in concepts}
    cavs = {}
    for concept in concepts:
        cav = CAV.load_cav(cav_paths[concept])

        cav_dir = cav.get_direction(cav.concepts[0]).astype("float32")
        cav_dir = cav_dir.reshape((cav_dir.shape[0], 1, 1))
        cavs[concept] = cav_dir
    return cavs


def plot_single_comparison(img1, img2):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(img1)
    axes[1].imshow(img2)
    for ax in axes:
        #ax.axis("off")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    return fig, axes


def decoder_forward_pre_upsampling(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        return h


def decoder_forward_from_upsampling(self, h):
    # timestep embedding
    temb = None

    # upsampling
    for i_level in reversed(range(self.num_resolutions)):
        for i_block in range(self.num_res_blocks + 1):
            h = self.up[i_level].block[i_block](h, temb)
            if len(self.up[i_level].attn) > 0:
                h = self.up[i_level].attn[i_block](h)
        if i_level != 0:
            h = self.up[i_level].upsample(h)

    # end
    if self.give_pre_end:
        return h

    h = self.norm_out(h)
    h = nonlinearity(h)
    h = self.conv_out(h)
    #if self.tanh_out:
    #    h = torch.tanh(h)
    return h


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


if __name__ == "__main__":
    parser = get_parser()
    main(parser.parse_args())
