import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from skimage import io
from skimage.color import label2rgb, rgb2gray
from skimage.segmentation import mark_boundaries, slic, felzenszwalb
from skimage.util import img_as_float, img_as_ubyte

from depth_predict.infer import Predictor
import matplotlib.pyplot as plt
import pickle


def segmentation(image: np.ndarray):
    segments = felzenszwalb(image, scale=200,
                            sigma=0.8, min_size=int(image.shape[0] * image.shape[1] / 200))
    labels, areas = np.unique(segments, return_counts=True)
    return {
        'seg': segments,
        'label': labels,
        'area': areas,
    }


def show_info(info: Dict):

    imgs = [
        ('Input', info['image'])
    ]

    if 'depth' in info.keys():
        imgs.append(('Depth', info['depth']))

    if 'seg' in info.keys():
        imgs.append(('Segmentation', info['seg']))

    plt.figure(figsize=(10, 5))
    for i, (title, img) in enumerate(imgs, 1):
        plt.subplot(1, len(imgs), i)
        plt.title(title)
        plt.imshow(img)

    plt.tight_layout()
    plt.show(block=True)


def save_info(info: Dict, input_path: Path, output_dir: Path):
    output_path = output_dir / (input_path.stem + '.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(info, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    parser.add_argument('--output_dir', default='outputs')
    parser.add_argument('--top_k', default=float('inf'), type=float)
    parser.add_argument('--weight_dir', default='weights')
    parser.add_argument('--show', action='store_true', default=False)
    parser.add_argument('--depth', action='store_true', default=False)
    parser.add_argument('--seg', action='store_true', default=False)
    args = parser.parse_args()

    image_paths = sorted(Path(args.input_dir).glob('*'))
    num_images = len(image_paths)

    if num_images == 0:
        print('No images.')
        exit(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    depth_predictor = None
    if not args.seg:
        weight_dir = Path(args.weight_dir)
        depth_predictor = Predictor(weight_dir / 'midas_v21_small-70d6b9c8.pt')

    for i, image_path in enumerate(image_paths):
        print(f'{i+1}/{num_images}: {image_path}')
        image = io.imread(image_path)
        info = {}
        info['image_path'] = str(image_path.resolve())
        info['image'] = image.copy()                        # 0-255 (H, W, C)
        image = img_as_float(image)

        if args.depth or args.seg:
            # debug mode
            if args.depth:
                info['depth'] = depth_predictor.predict(image)      # 0-255 (H, W)
            if args.seg:
                info.update(segmentation(image))                    # 0-255 (H, W)
            show_info(info)

        else:
            info['depth'] = depth_predictor.predict(image)      # 0-255 (H, W)
            info.update(segmentation(image))                    # 0-255 (H, W)
            save_info(info, image_path, output_dir)
            if args.show:
                show_info(info)

        if i + 1 > args.top_k:
            break


if __name__ == '__main__':
    main()
