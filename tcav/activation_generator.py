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
import itertools
import json
import numpy as np
from PIL import Image
import torch
import tensorflow as tf
from pathlib import Path
from torchvision.transforms.functional import normalize, resize
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using {} device: {}".format(device, torch.cuda.current_device()))
else:
    device = torch.device("cpu")
    print("Using {}".format(device))


class ActivationGenerator:
    """Activation generator for a basic image model"""
    def __init__(self, model, source_json, acts_dir, dataset_class, max_examples=500):
        """
        source_json (str): Path to a .json with filepaths for each img, categorised by concept
        dataset_class (Dataset): A dataset class to use in
        """
        self.model = model
        self.source_json = Path(source_json)
        self.acts_dir = Path(acts_dir)
        self.dataset_class = dataset_class
        self.max_examples = max_examples
        self.shape = model.shape

        with open(self.source_json, "r") as fp:
            self.concept_dict = json.load(fp)

        # Reduce size of each concept to max_examples
        self.concept_dict = {
            k: dict(itertools.islice(v.items(), self.max_examples)) for k, v in self.concept_dict.items()
        }

    def get_model(self):
        return self.model

    def get_activations_for_examples(self, examples, bottleneck):
        out_ = self.model(examples.to(device))
        del out_
        acts = self.model.bottlenecks_tensors[bottleneck]
        return acts.squeeze()

    def get_activations_for_concept(self, concept, bottleneck_names, batch_size=32, shuffle=True):
        """Get's activations for specified bottlenecks as np.arrays with no gradients"""
        dataset = self.dataset_class(self.concept_dict, concept)
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size, shuffle=shuffle)
        bns = {bn: [] for bn in bottleneck_names}
        with torch.no_grad():
            for sample in dataloader:
                out_ = self.model(sample["img"].to(device))
                for bn in bottleneck_names:
                    bns[bn].append(self.model.bottlenecks_tensors[bn].cpu().detach().numpy().squeeze())
                del out_

        bns = {k: np.concatenate(v) for k, v in bns.items()}
        return bns

    def process_and_load_activations(self, bottleneck_names, concepts):
        """Load activations if they exist, otherwise run imgs through model to create them and save as np.arrays"""
        acts = {}
        self.acts_dir.mkdir(exist_ok=True, parents=True)
        self.model.model.to(device)

        for concept in concepts:
            if concept not in acts:
                acts[concept] = {}
            act_paths = [self.acts_dir / f"acts_{concept}_{bottleneck_name}" for bottleneck_name in bottleneck_names]
            acts_exist = [path.exists() for path in act_paths]

            # If all the activations exist as a file then load, otherwise get from model
            if all(acts_exist):
                for i, bn in enumerate(bottleneck_names):
                    acts[concept][bn] = np.load(str(act_paths[i]), allow_pickle=True).squeeze()
            else:
                acts[concept] = self.get_activations_for_concept(concept, bottleneck_names)
                for i, bn in enumerate(bottleneck_names):
                    np.save(str(act_paths[i]), acts[concept][bn], allow_pickle=False)
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
