"""Compute depth maps for images in the input folder.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import Compose

from . import utils
from .midasnet import MidasNet_small
from .transform import NormalizeImage, PrepareForNet, Resize


def load_model_and_transform(weight_path: Path) -> nn.Module:
    model = MidasNet_small(weight_path, features=64, backbone="efficientnet_lite3",
                           exportable=True, non_negative=True, blocks={'expand': True})
    model = model.eval()

    # load network
    net_w, net_h = 256, 256
    resize_mode = "upper_bound"
    normalization = NormalizeImage(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    return model, transform


class Predictor:

    def __init__(self, weight_path: Path):
        self.model, self.transform = load_model_and_transform(weight_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.bits = 1

    @torch.no_grad()
    def predict(self, image: np.ndarray):
        img_input = self.transform({"image": image})["image"]
        sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
        prediction = self.model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
        prediction = utils.normalize_depth_image(prediction, bits=self.bits)
        return prediction


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--weight_path',
                        default='weights/midas_v21_small-70d6b9c8.pt',
                        help='path to the trained weights of model'
                        )
    parser.add_argument('--top_k', default=10**6, type=int,
                        help='Run on top_k images in input_dir')
    args = parser.parse_args()

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    if args.input_dir is None and args.image is None:
        print('--input_dir or --image must be specified')
        exit(-1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    predictor = Predictor(args.weight_path)

    if args.image is not None:
        input_paths = [Path(args.image)]
    else:
        input_dir = Path(args.input_dir)
        input_paths = sorted(input_dir.glob('*'))

    num_images = len(input_paths)
    with torch.no_grad():
        for i, input_path in enumerate(input_paths):
            print("[{:05d}/{:05d}] {}".format(i + 1, num_images, input_path))
            image = utils.read_image(str(input_path))
            prediction = predictor.predict(image)
            output_path = (output_dir / input_path.name).with_suffix('.png')
            utils.write_depth(output_path, prediction, pfm=False, bits=1)

        print("finished")
