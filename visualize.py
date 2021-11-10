from pathlib import Path
import json
from PIL import Image, ImageDraw
import numpy as np


class Visualizer:

    def __init__(self) -> None:
        pass

    def visualize(self,
                  json_dict,
                  image_path: Path,
                  output_path: Path):
        image = Image.open(image_path).convert('RGB')
        drawer = ImageDraw.Draw(image, mode='RGBA')
        for word_box in json_dict['word_boxes']:
            bbox = np.array(word_box['bbox'], dtype=np.float32)
            drawer.polygon(bbox.flatten().tolist(), fill=(255, 0, 0, 50), outline='red')

        image.save(output_path, 'png')


def main():
    input_dir = Path('output_images')
    output_dir = Path('output_visualize')
    output_dir.mkdir(exist_ok=True, parents=True)

    visualizer = Visualizer()
    for i, json_path in enumerate((input_dir / "labels").glob('*.json')):
        with open(json_path, 'rt') as f:
            json_dict = json.load(f)
        image_path = input_dir / "images" / json_dict['image_name']
        output_path = output_dir / image_path.name
        visualizer.visualize(json_dict, image_path, output_path)
        print(f'[{i+1:09d}] {output_path}')

        if i == 5:
            break # debug


if __name__ == '__main__':
    main()
