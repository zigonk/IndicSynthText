import argparse
from skimage import io
from pathlib import Path

import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('depth_output_dir')
    parser.add_argument('seg_output_dir')
    parser.add_argument('--output_dir', default='outputs')
    args = parser.parse_args()

    depth_output_dir = Path(args.depth_output_dir)
    seg_output_dir = Path(args.seg_output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    depth_imgs = sorted(depth_output_dir.glob('*.png'))
    seg_pkls = sorted(seg_output_dir.glob('*.pkl'))
    assert len(depth_imgs) == len(seg_pkls), f'{len(depth_imgs)} != {len(seg_pkls)}'

    num_images = len(depth_imgs)

    if num_images == 0:
        print('No images.')
        exit(0)

    for i, (depth_img_path, seg_pkl) in enumerate(zip(depth_imgs, seg_pkls)):
        assert depth_img_path.stem == seg_pkl.stem, f'{depth_img_path.stem} != {seg_pkl.stem}'

        print(f'{i+1}/{num_images}: {depth_img_path}')
        depth_image = io.imread(str(depth_img_path), as_gray=True)
        with open(seg_pkl, 'rb') as f:
            seg_info = pickle.load(f)

        info = {}
        info['image_path'] = seg_info['image_path']
        info['image'] = seg_info['image']
        info['depth'] = depth_image
        info['seg'] = seg_info['seg']
        info['label'] = seg_info['label']
        info['area'] = seg_info['area']

        output_path = output_dir / (depth_img_path.stem + '.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(info, f)


if __name__ == '__main__':
    main()
