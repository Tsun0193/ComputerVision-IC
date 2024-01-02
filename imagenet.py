import os

import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.models import (
    AlexNet_Weights,
    ConvNeXt_Tiny_Weights,
    ResNet152_Weights,
    VGG19_Weights,
    alexnet,
    convnext_tiny,
    resnet152,
    vgg19,
)

from config import BASE_URL, BATCH_SIZE, LOC_SYNSET_MAPPING, LOC_VAL_SOLUTION

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IMG_Dataset(Dataset):
    """Image Dataset class"""

    def __init__(self, image_list, base_url, transform=None):
        """Constructor of the class

        Args:
            image_list (list(str)): List of image file name.
            base_url (string): Base url of image file.
            transform (function): A function to transform input image to another format.
        """
        self.transform = transform
        self.images = image_list
        self.base_url = base_url

    def __len__(self):
        """Get length of image list function

        Returns:
            Length of images list
        """

        return len(self.images)

    def __getitem__(self, index):
        """Get item function

        Args:
            index (int): Index of an image.

        Returns:
            Three dimension matrix of image.
        """

        img_path = os.path.join(self.base_url, self.images[index])
        image = read_image(img_path, mode=torchvision.io.ImageReadMode.RGB)
        if self.transform is not None:
            try:
                transformed_img = self.transform(image)
                image = transformed_img
            except:
                print(f"error found at file {img_path}")
                print(image.shape)
        return image


def smap(prediction_string, mapping):
    """Mapping from prediction string (ground truth) in imagenet dataset to index of label that widely use in general

    Args:
        prediction_string (str): a imagenet encode format for image classes.
        mapping (dict): a dictionary mapping between prediction string in imagenet dataset and index of image classes in pytorch pretrained models for imagenet1k-dataset.

    Returns:
        An index of index class in pytorch pretrained models for imagenet1k-dataset
    """

    key = prediction_string.split(" ")[0]
    idx = mapping[key]

    return idx


def get_loader(
    image_list, batch_size, transform, base_url, num_workers=4, pin_memory=True
):
    """Load images function

    Args:
        image_list (list(str)): List of image file name.
        batch_size (int): Number of images load in a batch.
        transform (function): A function to transform input image to another format.
        base_url (str): Base url of image file.
        num_workers (int): how many subprocesses to use for data loading. Default: 4.
        pin_memory (bool): If True, the data loader will copy Tensors into device/CUDA pinned memory before return them.

    Returns:
        An iterable over given image list.
    """

    test_dataset = IMG_Dataset(image_list, base_url, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )
    return test_loader


def get_prediction(model, data, result_path):
    """Get prediction function

    Args:
        model (model): A model to use.
        data (DataLoader): An iterable object of input images.
        result_path (str): An path location to store prediction.
    """

    print(f"Start predict {result_path.split('/')[1].split('.')[0].split('_')[0]}")
    with torch.no_grad():
        preds = torch.empty((0, 5), dtype=torch.int64).to(device)
        for batch_idx, batch_data in enumerate(data):
            batch_preds = model(batch_data.to(device)).softmax(0)
            idx = batch_preds.argsort(dim=1, stable=True, descending=True)
            idx = idx[:, :5]
            preds = torch.cat((preds, idx), dim=0)

        result = pd.DataFrame(
            preds.cpu().numpy(), columns=["Top1", "Top2", "Top3", "Top4", "Top5"]
        )
        result["Image"] = val_dataset["ImageName"]
        result.to_csv(result_path, index=False)


def assess_result(result_path):
    """Assess result function (This function print out result of some metrics for model prediction)

    Args:
        result_path: path location of model predictions for test dataset.
    """

    model_name = result_path.split("/")[-1].split("_")[0]
    result = pd.read_csv(result_path)
    cnt = 0
    for i in range(len(ground_truth)):
        if ground_truth.loc[i, "PredictionCategories"] == result.loc[i, "Top1"]:
            cnt += 1
    print(
        f"Top-1 accuracy of {model_name} model for validation test is:",
        f"{cnt / len(ground_truth) * 100}%",
    )
    cnt = 0
    for i in range(len(ground_truth)):
        if (
            ground_truth.loc[i, "PredictionCategories"] == result.loc[i, "Top1"]
            or ground_truth.loc[i, "PredictionCategories"] == result.loc[i, "Top2"]
            or ground_truth.loc[i, "PredictionCategories"] == result.loc[i, "Top3"]
            or ground_truth.loc[i, "PredictionCategories"] == result.loc[i, "Top4"]
            or ground_truth.loc[i, "PredictionCategories"] == result.loc[i, "Top5"]
        ):
            cnt += 1
    print(
        f"Top-5 accuracy of {model_name} model for validation test is:",
        f"{cnt / len(ground_truth) * 100}%",
    )


if __name__ == "__main__":
    # Get dictionary between encoded labels and real labels of imagenet dataset
    with open(LOC_SYNSET_MAPPING, "r") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    dic = {}
    for line in lines:
        tmp = line.split()
        dic[tmp[0]] = " ".join(tmp[1:]).split(", ")

    # Because pretrained models use the same meta["categories"] for encode 1000 labels
    # so I use same mapping between id in meta and encoded categories in imagenetdataset
    mapping = {}
    for key in dic:
        for i in range(len(AlexNet_Weights.IMAGENET1K_V1.meta["categories"])):
            if AlexNet_Weights.IMAGENET1K_V1.meta["categories"][i] in dic[key]:
                mapping[key] = i
                break

    # Build validation dataset
    val_dataset = pd.read_csv(LOC_VAL_SOLUTION)
    val_dataset["PredictionCategories"] = val_dataset["PredictionString"].apply(
        lambda x: smap(x, mapping)
    )
    val_dataset = val_dataset.drop(columns=["PredictionString"])
    val_dataset["ImageName"] = val_dataset["ImageId"].apply(lambda x: x + ".JPEG")

    # Load pretrained model
    convnext_weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
    convnext_model = convnext_tiny(weights=convnext_weights).to(device)
    alexnet_weights = AlexNet_Weights.IMAGENET1K_V1
    alexnet_model = alexnet(weights=alexnet_weights).to(device)
    resnet_weights = ResNet152_Weights.IMAGENET1K_V1
    resnet_model = resnet152(weights=resnet_weights).to(device)
    vgg_weights = VGG19_Weights.IMAGENET1K_V1
    vgg_model = vgg19(weights=vgg_weights).to(device)

    # Load data to each pretrained model
    alexnet_transform = alexnet_weights.transforms()
    convnext_transform = convnext_weights.transforms()
    resnet_transform = resnet_weights.transforms()
    vgg_transform = vgg_weights.transforms()

    alexnet_data = get_loader(
        val_dataset["ImageName"].tolist(), BATCH_SIZE, alexnet_transform, BASE_URL
    )
    convnext_data = get_loader(
        val_dataset["ImageName"].tolist(), BATCH_SIZE, convnext_transform, BASE_URL
    )
    resnet_data = get_loader(
        val_dataset["ImageName"].tolist(), BATCH_SIZE, resnet_transform, BASE_URL
    )
    vgg_data = get_loader(
        val_dataset["ImageName"].tolist(), BATCH_SIZE, vgg_transform, BASE_URL
    )

    # Get prediction of pretrained models
    # I store result of all pretrained model to reduce running time in after experiments
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("results/alexnet_result.csv"):
        get_prediction(alexnet_model, alexnet_data, "results/alexnet_result.csv")
    if not os.path.exists("results/convnext_result.csv"):
        get_prediction(convnext_model, convnext_data, "results/convnext_result.csv")
    if not os.path.exists("results/restnet_result.csv"):
        get_prediction(resnet_model, resnet_data, "results/resnet_result.csv")
    if not os.path.exists("results/vgg_result.csv"):
        get_prediction(vgg_model, vgg_data, "results/vgg_result.csv")

    # Assess result
    ground_truth = val_dataset[["ImageName", "PredictionCategories"]]
    assess_result("results/alexnet_result.csv")
    assess_result("results/convnext_result.csv")
    assess_result("results/resnet_result.csv")
    assess_result("results/vgg_result.csv")
