
from pathlib import Path


import numpy as np
import torch
import gradio as gr

from tcav.utils import device
from torchvision.transforms.transforms import ToTensor
from perturb_image_generation import load_cavs, decoder_forward_pre_upsampling, decoder_forward_from_upsampling, undo_preprocess, preprocess
from run_tcav import create_model
import torch.nn.functional as F


def main():
    # model_path = "/home/lina3782/dev/stable-diffusion/logs/2023-05-17T17-45-44_autoencoder_kl_16x16x16_element/checkpoints/epoch=000046.ckpt"
    # ldm_config_path = "/home/lina3782/dev/stable-diffusion/logs/2023-05-17T17-45-44_autoencoder_kl_16x16x16_element/configs/2023-05-17T17-45-44-project.yaml"
    # cav_path = "/mnt/data01/lina3782/element/tcav/simple_all/ae/maxpool/cavs"
    # concepts = ["square", "circle", "triangle", "plus", "red", "green", "blue", "solid", "spots_polka", "stripes_diagonal"]

    # model_path = "/home/lina3782/labs/explain/imagenet/models/vqgan/vqgan_imagenet_f16_1024/last.ckpt"
    # ldm_config_path = "/home/lina3782/labs/explain/imagenet/models/vqgan/vqgan_imagenet_f16_1024/model.yaml"
    # cav_path = "/mnt/data01/lina3782/imagenet/tcav/vqgan/f16/maxpool/cavs"
    # concepts = ["striped", "polka", "dotted", "lined", "banded", "meshed", "zigzagged"]

    model_path = "/home/lina3782/dev/stable-diffusion/logs/2023-06-19T11-58-56_autoencoder_kl_16x16x16_element_non_overlapping/checkpoints/epoch=000060.ckpt"
    ldm_config_path = "/home/lina3782/dev/stable-diffusion/logs/2023-06-19T11-58-56_autoencoder_kl_16x16x16_element_non_overlapping/configs/2023-06-19T11-58-56-project.yaml"
    cav_path = "/mnt/data01/lina3782/element/tcav/simple_all_non_overlapping/ae/maxpool/cavs"
    concepts = ["square", "circle", "triangle", "plus", "red", "green", "blue", "solid", "spots_polka", "stripes_diagonal"]

    bottleneck_layer = "decoder.mid.block_2"
    share = True

    cavs = load_cavs(concepts, cav_path, bottleneck_layer)
    cavs = convert_cavs_to_torch(cavs)

    model = create_model(path=model_path, ldm_config_path=ldm_config_path)
    model.to(device)
    model.eval()
    to_tensor = ToTensor()

    def perturb_image(img, mask, value, up_concepts, down_concepts):
        img = to_tensor(img).unsqueeze(0)
        #img = preprocess(img)
        acts = get_activations(model, img.to(device))
        mask = to_tensor(mask["mask"])[:1].unsqueeze(0)
        mask = F.interpolate(mask, size=acts.shape[-2:]).to(device)

        up_perturbation = combine_cavs(up_concepts, cavs)
        down_perturbation = combine_cavs(down_concepts, cavs)
        perturbation = value * (up_perturbation - down_perturbation)
        print(perturbation.shape)
        print(up_perturbation.shape)

        acts += mask * perturbation
        out = get_output_from_acts(model, acts).cpu()
        #out = undo_preprocess(out)
        out = out[0].permute(1, 2, 0)
        out[out < 0] = 0
        out[out > 1] = 1
        return np.array(out)

    with gr.Blocks() as demo:
        gr.Markdown("Perturb images from the Element dataset")
        perturb_value = gr.Slider(
            0,
            100,
            value=10,
            label="Perturbation Strength",
            info="How large should the perturbation be?"
        )
        concepts_to_increase = gr.CheckboxGroup(
            concepts,
            label="Concepts",
            info="Which concepts do you want to increase?"
        )
        concepts_to_decrease = gr.CheckboxGroup(
            concepts,
            label="Concepts",
            info="Which concepts do you want to decrease?"
        )
        with gr.Row():
            mask_input = gr.ImageMask()
            image_input = gr.Image()
            image_output = gr.Image()
        image_button = gr.Button("Perturb")

        image_button.click(perturb_image, inputs=[image_input, mask_input, perturb_value, concepts_to_increase, concepts_to_decrease], outputs=image_output)

    demo.launch(share=share)

    print("Done!")




def combine_cavs(concept_list, cavs):
    cav_list = [cavs[concept] for concept in concept_list]
    if len(concept_list) > 0:
        out = torch.stack(cav_list, dim=0)
        out = torch.sum(out, dim=0)
    else:
        example_cav = next(iter(cavs.values()))
        out = torch.zeros_like(example_cav)
    return out


def convert_cavs_to_torch(cavs):
    for k, v in cavs.items():
        cavs[k] = torch.tensor(v).to(device)
    return cavs


def get_activations(model, images):
    with torch.no_grad():
        posterior = model.encode(images)
        z = posterior.sample() #[0]
        z_post_quant = model.post_quant_conv(z)
        activations = decoder_forward_pre_upsampling(model.decoder, z_post_quant)
    return activations


def get_output_from_acts(model, activations):
    with torch.no_grad():
        output = decoder_forward_from_upsampling(
            model.decoder,
            activations
        )
    return output


def greet(name):
    return "Hello " + name + "!"


if __name__ == "__main__":
    main()

