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

from typing import Tuple, Sequence
import copy
from torch.autograd import grad

import torch
from torchvision import models


class ModelWrapper:
    """Model wrapper to hold pytorch image models and set up the needed
    hooks to access the activations and grads.
    """

    def __init__(
        self, model: torch.nn.Module, bottlenecks: dict, labels: Sequence[str]
    ):
        """Initialize wrapper with model and set up the hooks to the bottlenecks.
        Args:
            model (nn.Module): Model to test
            bottlenecks (dict): Dictionary attaching names to the layers to hook into. Expects, at least, an input,
                logit and prediction.
            labels (list): Class labels in order the model expects
        """
        self.ends = None
        self.y_input = None
        self.loss = None
        self.bottlenecks_gradients = None
        self.bottlenecks_tensors = {}
        self.model = copy.deepcopy(model)
        self.model.eval()
        self.shape = (224, 224)
        self.labels = labels

        def save_activation(name):
            """Creates hooks to the activations
            Args:
                name (string): Name of the layer to hook into
            """

            def hook(module, input, output):
                """Saves the activation hook to dictionary"""
                self.bottlenecks_tensors[name] = output

            return hook

        for name, mod in self.model.named_modules():
            if name in bottlenecks.keys():
                mod.register_forward_hook(save_activation(bottlenecks[name]))

    def _make_gradient_tensors(
        self, x: torch.Tensor, y: int, bottleneck_name: str
    ) -> torch.Tensor:
        """
        Makes gradient tensor for logit y w.r.t. layer with activations

        Args:
            x (tensor): Model input
            y (int): Index of logit (class)
            bottleneck_name (string): Name of layer activations
        Returns:
            (torch.tensor): Gradients of logit w.r.t. to activations
        """
        out = self.model(x.unsqueeze(0))
        acts = self.bottlenecks_tensors[bottleneck_name]
        return grad(out[:, y], acts)[0]

    def eval(self):
        """Sets wrapped model to eval mode."""
        self.model.eval()

    def train(self):
        """Sets wrapped model to train mode."""
        self.model.train()

    def __call__(self, x: torch.Tensor):
        """Calls prediction on wrapped model."""
        self.ends = self.model(x)
        return self.ends

    def get_gradient(
        self, x: torch.Tensor, y: int, bottleneck_name: str
    ) -> torch.Tensor:
        """Returns the gradient at a given bottle_neck.
        Args:
            x: Model input
            y: Index of the logit layer (class)
            bottleneck_name: Name of the bottleneck to get gradients w.r.t.
        Returns:
            (torch.tensor): Tensor containing the gradients at layer.
        """
        self.y_input = y
        return self._make_gradient_tensors(x, y, bottleneck_name)

    def id_to_label(self, idx):
        return self.labels[idx]

    def label_to_id(self, label):
        return self.labels.index(label)


def create_model(freeze_weights=False, n_classes=13):
    model = models.resnet50(pretrained=True)

    if freeze_weights:
        for param in model.parameters():
            param.requires_grad = False

    # NB: Newly initialised layers have requires_grad=True
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5), torch.nn.Linear(2048, n_classes)
    )
    return model
