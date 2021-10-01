"""Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from multiprocessing import dummy as multiprocessing
import os
import os.path
import numpy as np
from PIL import Image
import torch
import tensorflow as tf
from pathlib import Path
from torchvision.transforms.functional import normalize, resize

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using {} device: {}".format(device, torch.cuda.current_device()))
else:
    device = torch.device("cpu")
    print("Using {}".format(device))


class ActivationGenerator():
    """Activation generator for a basic image model"""
    def __init__(self, model, source_dir, acts_dir, max_examples=500):
        self.model = model
        self.source_dir = Path(source_dir)
        self.acts_dir = Path(acts_dir)
        self.max_examples = max_examples
        self.shape = model.shape

    def get_model(self):
        return self.model

    def get_activations_for_examples(self, examples, bottleneck):
        out_ = self.model(examples.to(device))
        del out_
        acts = self.model.bottlenecks_tensors[bottleneck]
        return acts.squeeze()

    def process_and_load_activations(self, bottleneck_names, concepts):
        # TODO: Need to replace this with datasets and dataloaders
        #  Can then do-away with separate folders for concepts
        #  instead can use .jsons to say which is which
        acts = {}
        self.acts_dir.mkdir(exist_ok=True, parents=True)
        self.model.model.to(device)
        self.model.eval()

        for concept in concepts:
            if concept not in acts:
                acts[concept] = {}
            act_paths = [self.acts_dir / f"acts_{concept}_{bottleneck_name}" for bottleneck_name in bottleneck_names]
            acts_exist = [path.exists() for path in act_paths]

            # Must run examples through model so that hooks are generated
            if not all(acts_exist):
                examples = self.get_examples_for_concept(concept).to(device)
                out = self.model(examples)
                del out

            # If the activations exist as a file then load
            # otherwise get from model
            # Could remove loading from file if generated anyway?
            for i, bottleneck_name in enumerate(bottleneck_names):
                if acts_exist[i]:
                    with tf.io.gfile.GFile(act_paths[i], 'rb') as f:
                        acts[concept][bottleneck_name] = np.load(f, allow_pickle=True).squeeze()
                else:
                    acts[concept][bottleneck_name] = self.model.bottlenecks_tensors[bottleneck_name].cpu().detach().numpy().squeeze()
                    with tf.io.gfile.GFile(act_paths[i], 'w') as f:
                        np.save(f, acts[concept][bottleneck_name], allow_pickle=False)
        return acts

    def get_examples_for_concept(self, concept):
        concept_dir = self.source_dir / concept
        img_paths = [
          concept_dir / d for d in tf.io.gfile.listdir(concept_dir)
        ]
        imgs = self.load_images_from_files(
          img_paths, self.max_examples)
        return imgs

    def load_image_from_file(self, path):
        img = Image.open(path)
        img = np.array(img).astype("float32") / 255
        img = np.expand_dims(img, 0)
        img = np.repeat(img, repeats=3, axis=0)
        img = torch.from_numpy(img)

        # As per pretrained pytorch models
        img = normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = resize(img, self.shape)

        return img

    def load_images_from_files(self,
                               filenames,
                               max_imgs=500,
                               do_shuffle=True,
                               run_parallel=True,
                               num_workers=50):
        """Return image arrays from filenames.

        Args:
          filenames: locations of image files.
          max_imgs: maximum number of images from filenames.
          do_shuffle: before getting max_imgs files, shuffle the names or not
          run_parallel: get images in parallel or not
          shape: desired shape of the image
          num_workers: number of workers in parallelization.

        Returns:
          image arrays

        """
        imgs = []
        # First shuffle a copy of the filenames.
        filenames = filenames[:]
        if do_shuffle:
            np.random.shuffle(filenames)

        if run_parallel:
            pool = multiprocessing.Pool(num_workers)
            imgs = pool.map(
                lambda filename: self.load_image_from_file(filename),
                filenames[:max_imgs])
            pool.close()
            imgs = [img for img in imgs if img is not None]
            if len(imgs) <= 1:
                raise ValueError(
                  'You must have more than 1 image in each class to run TCAV.')
        else:
            for filename in filenames:
                img = self.load_image_from_file(filename)
                if img is not None:
                    imgs.append(img)
                if len(imgs) >= max_imgs:
                    break
            if len(imgs) <= 1:
                raise ValueError(
                  'You must have more than 1 image in each class to run TCAV.')
        return torch.stack(imgs, dim=0)
