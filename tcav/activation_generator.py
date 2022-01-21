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

import itertools
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using {} device: {}".format(device, torch.cuda.current_device()))
else:
    device = torch.device("cpu")
    print("Using {}".format(device))


class ActivationGenerator:
    """Activation generator for a basic image model"""
    def __init__(self, model, source_json, acts_dir, dataset_class, max_examples=500, num_workers=0, prefix=""):
        """
        source_json (str): Path to a .json with filepaths for each img, categorised by concept
        dataset_class (Dataset): A dataset Class. Requires inputs:
                                 json_file (img file paths and metadata),
                                 split (which data split to use),
                                 prefix (filepath prefixes)
        """
        self.model = model
        self.model.model.to(device)
        self.prefix = prefix
        self.acts_dir = Path(acts_dir)
        self.dataset_class = dataset_class
        self.max_examples = max_examples
        self.shape = model.shape
        self.num_workers = num_workers

    def get_model(self):
        return self.model

    def get_activations_for_examples(self, examples, bottleneck, batch_size=32, grad=False):
        acts = []
        if grad:
            for batch in torch.split(examples, batch_size):
                out_ = self.model(batch.to(device))
                del out_
                acts.append(self.model.bottlenecks_tensors[bottleneck].cpu())
        else:
            with torch.no_grad():
                for batch in torch.split(examples, batch_size):
                    out_ = self.model(batch.to(device))
                    del out_
                    acts.append(self.model.bottlenecks_tensors[bottleneck].cpu())
        return torch.cat(acts)

    def get_activations_for_concept(self, concept, bottleneck_names, batch_size=32, shuffle=True, n_repeats=1):
        """Get's activations for specified bottlenecks as np.arrays with no gradients"""
        dataset = self.dataset_class(concept, prefix=self.prefix)
        self.model.eval()
        bns = {bn: [] for bn in bottleneck_names}
        for i in range(n_repeats):
            dataloader = DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=self.num_workers)
            with torch.no_grad():
                for sample in dataloader:
                    out_ = self.model(sample[0].to(device))
                    for bn in bottleneck_names:
                        bns[bn].append(self.model.bottlenecks_tensors[bn].cpu().detach().numpy())
                    del out_

        bns = {k: np.concatenate(v) for k, v in bns.items()}
        return bns

    def process_and_load_activations(self, bottleneck_names, concepts, overwrite=False, n_repeats=1):
        """Load activations if they exist, otherwise run imgs through model to create them and save as np.arrays

        bottleneck_names: Names of bottleneck layers to load activations for as in model.bottlenecks
        concepts: Concepts to load activations for
        overwrite: Whether to overwrite activations even if they all exist - activations overwritten either way if any do not exist
        n_repeats: number of times to load dataset (for use with augmentations)
        """
        acts = {}
        self.acts_dir.mkdir(exist_ok=True, parents=True)
        self.model.model.to(device)

        for concept in concepts:
            if concept not in acts:
                acts[concept] = {}
            act_paths = [self.acts_dir / f"acts_{concept}_{bottleneck_name}.npy" for bottleneck_name in bottleneck_names]
            acts_exist = [path.exists() for path in act_paths]

            # If all the activations exist as a file then load, otherwise get from model
            if all(acts_exist) and not overwrite:
                for i, bn in enumerate(bottleneck_names):
                    acts[concept][bn] = np.load(str(act_paths[i]), allow_pickle=True).squeeze()
            else:
                # The dataset is not shuffled for reproducibility
                # This means the dataset will be loaded in the order it appear in .json
                # Ensure that this is acceptable (imgs are not ordered in some way)
                acts[concept] = self.get_activations_for_concept(concept, bottleneck_names, shuffle=False, n_repeats=n_repeats)
                for i, bn in enumerate(bottleneck_names):
                    np.save(str(act_paths[i]), acts[concept][bn], allow_pickle=False)
        return acts

    def get_examples_for_concept(self, concept, n=None, shuffle=False, return_ids=False):
        if n is None:
            n = self.max_examples
        dataset = self.dataset_class(concept, prefix=self.prefix)
        dataloader = DataLoader(dataset, n, shuffle=shuffle, num_workers=self.num_workers)
        for sample in dataloader:
            imgs = sample[0]
            if return_ids:
                ids = sample[1]
            break

        if return_ids:
            return ids, imgs
        else:
            return imgs
