import argparse
from pathlib import Path
from typing import Dict
import cv2

import numpy as np
from skimage import io
from skimage.color import label2rgb, rgb2gray
from skimage.segmentation import mark_boundaries, slic, felzenszwalb
from skimage.util import img_as_float, img_as_ubyte

import imutils
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
    parser.add_argument('--output_dir', default='outputs_seg')
    parser.add_argument('--vis_dir', default='outputs_seg_vis')
    parser.add_argument('--max_width', default=1920, type=int)
    parser.add_argument('--max_height', default=1080, type=int)
    parser.add_argument('--max_size', default=1024, type=int)
    parser.add_argument('--top_k', default=float('inf'), type=float)
    parser.add_argument('--show', action='store_true', default=False)
    args = parser.parse_args()

    image_paths = sorted(Path(args.input_dir).glob('*'))
    num_images = len(image_paths)

    if num_images == 0:
        print('No images.')
        exit(0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    vis_dir = Path(args.vis_dir)
    vis_dir.mkdir(exist_ok=True, parents=True)

    for i, image_path in enumerate(image_paths):
        print(f'{i+1}/{num_images}: {image_path}')
        image = io.imread(image_path)
        h, w = image.shape[:2]
        if h > w and h > args.max_height:
            image = imutils.resize(image, height=args.max_height, inter=cv2.INTER_LINEAR)
        elif w > h and w > args.max_width:
            image = imutils.resize(image, width=args.max_width, inter=cv2.INTER_LINEAR)
        elif h == w and h > args.max_size:
            image = imutils.resize(image, height=args.max_size, inter=cv2.INTER_LINEAR)
        info = {}
        info['image_path'] = image_path.name
        info['image'] = image.copy()                        # 0-255 (H, W, C)
        image = img_as_float(image)
        info.update(segmentation(image))                          # 0-255 (H, W)
        save_info(info, image_path, output_dir)
        if args.show:
            show_info(info)

        vis_image = label2rgb(info['seg'], image, alpha=0.8, bg_label=0)
        vis_image = img_as_ubyte(vis_image)
        io.imsave(str(vis_dir / image_path.name), vis_image)

        if i + 1 > args.top_k:
            break


if __name__ == '__main__':
    main()
