import argparse
from pathlib import Path
from typing import List

import cv2
import imutils
import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image_paths', type=Path, nargs='+')
    parser.add_argument('--output_dir', type=Path, default='outputs')
    parser.add_argument('--max_width', default=1920, type=int)
    parser.add_argument('--max_height', default=1080, type=int)
    parser.add_argument('--max_size', default=1024, type=int)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    image_paths: List[Path] = args.image_paths
    for image_path in tqdm.tqdm(image_paths):
        image = cv2.imread(str(image_path))
        h, w = image.shape[:2]
        if h > w and h > args.max_height:
            image = imutils.resize(image, height=args.max_height, inter=cv2.INTER_LINEAR)
        elif w > h and w > args.max_width:
            image = imutils.resize(image, width=args.max_width, inter=cv2.INTER_LINEAR)
        elif h == w and h > args.max_size:
            image = imutils.resize(image, height=args.max_size, inter=cv2.INTER_LINEAR)
        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), image)


if __name__ == '__main__':
    main()
