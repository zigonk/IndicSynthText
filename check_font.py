from typing import List
from PIL import Image, ImageFont, ImageDraw
from pathlib import Path
from argparse import ArgumentParser


def draw_char(font: ImageFont.FreeTypeFont, ch: str, rotation: float):
    ch_bb = list(font.getbbox(ch))
    ch_image = Image.new('RGB', (ch_bb[2], ch_bb[3]), (0, 0, 0))
    draw = ImageDraw.Draw(ch_image)
    draw.text((0, 0), ch, font=font, fill=(255, 255, 255))
    ch_image = ch_image.rotate(rotation, Image.BICUBIC, expand=True)
    return ch_image


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('font_paths', nargs='+', type=Path)
    args = parser.parse_args()

    word_text = u'v\u1EA5n'

    font_paths: List[Path] = args.font_paths
    for font_path in font_paths:
        print('Font:', font_path)
        font = ImageFont.truetype(str(font_path), 200)

        ascent, descent = font.getmetrics()
        print('metrics', ascent, descent)

        w, h = font.getsize(word_text)
        print('size', w, h)

        offsetx, offsety = font.getoffset(word_text)
        print('offset', offsetx, offsety)

        image = draw_char(font, word_text, 0)
        print('Image Size', image.size)

        out_path = font_path.with_suffix('.png')
        print('Out:', out_path)
        image.save(out_path)
