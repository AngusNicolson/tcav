import json
from typing import Union, Callable
from copy import deepcopy

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class JsonDataset(Dataset):
    """json dataset. Define dataset using a .json which splits the images by train/val/test or concept"""

    def __init__(
        self,
        json_file: Union[str, dict],
        split: str = "train",
        prefix: str = None,
        transform: Callable = None,
        load_img: bool = True,
    ):
        """
        Args:
            json_file (string): Path to the json metadata file.
            split (string): Which data split to load
            prefix (string, optional): Add a prefix to file paths.
            transform (callable, optional): Optional transform to be applied on a sample.
            load_img (bool, optional): Whether to load the image or just the metadata, default is to load
        """
        self.split = split
        self.load_img = load_img
        self.transform = transform
        self.metadata_labels = ["label"]

        if self.transform is None:
            self.transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    T.Resize((224, 224)),
                ]
            )

        if type(json_file) == dict:
            self.metadata = deepcopy(json_file)
        else:
            with open(json_file, "r") as fp:
                self.metadata = json.load(fp)

        if split is not None:
            self.metadata = self.metadata[split]

        if prefix is not None:
            for k in self.metadata.keys():
                self.metadata[k]["path"] = prefix + self.metadata[k]["path"]

        self.ids = list(self.metadata.keys())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id_metadata = self.metadata[self.ids[idx]]
        if self.load_img:
            img = self.load_img_(id_metadata["path"])
            img = self.transform(img)
        else:
            img = None

        sample = {"id": self.ids[idx], "img": img}
        sample[0] = sample["img"]
        sample.update({k: id_metadata[k] for k in self.metadata_labels})
        sample[1] = sample["label"]
        return sample

    @staticmethod
    def load_img_(path):
        with open(path, "rb") as fp:
            img = Image.open(fp)
            return img.convert("RGB")
