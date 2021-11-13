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

    depth_pkls = sorted(depth_output_dir.glob('*.pkl'))
    seg_pkls = sorted(seg_output_dir.glob('*.pkl'))
    assert len(depth_pkls) == len(seg_pkls), f'{len(depth_pkls)} != {len(seg_pkls)}'

    num_images = len(depth_pkls)

    if num_images == 0:
        print('No images.')
        exit(0)

    for i, (depth_pkl, seg_pkl) in enumerate(zip(depth_pkls, seg_pkls)):
        assert depth_pkl.stem == seg_pkl.stem, f'{depth_pkl.stem} != {seg_pkl.stem}'

        print(f'{i+1}/{num_images}: {depth_pkl}')
        with open(depth_pkl, 'rb') as f:
            depth_image = pickle.load(f)
        with open(seg_pkl, 'rb') as f:
            seg_info = pickle.load(f)

        info = {}
        info['image_path'] = seg_info['image_path']
        info['image'] = seg_info['image']
        info['depth'] = depth_image
        info['seg'] = seg_info['seg']
        info['label'] = seg_info['label']
        info['area'] = seg_info['area']

        output_path = output_dir / (depth_pkl.stem + '.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(info, f)


if __name__ == '__main__':
    main()
