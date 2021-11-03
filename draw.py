from PIL import Image, ImageFont, ImageDraw
import numpy as np


class BaselineState(object):
    A = [0.50, 0.05]

    def __init__(self, a) -> None:
        self.a = a

    def curve(self, x):
        return self.a*x**2

    def differential(self, x):
        return 2*self.a*x

    @staticmethod
    def get_sample():
        """
        Returns the functions for the curve and differential for a and b
        """
        sgn = 1.0
        if np.random.rand() < 0.5:
            sgn = -1

        a = BaselineState.A[1]*np.random.randn() + sgn*BaselineState.A[0]
        return BaselineState(a)


def draw_char(font: ImageFont.FreeTypeFont, ch: str, rotation: float):
    ch_bb = list(font.getbbox(ch))
    ch_image = Image.new('RGBA', (ch_bb[2], ch_bb[3]), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ch_image)
    draw.text((0, 0), ch, font=font, fill=(255, 255, 255, 255))
    ch_image = ch_image.rotate(rotation, Image.BICUBIC, expand=True)
    return ch_image


if __name__ == '__main__':

    text = 'habcddcbah\nasdasdfsd'
    wl = len(text)
    isword = len(text.split()) == 1
    font = ImageFont.truetype(
        'SynthTextGen/fonts/ubuntuen/Ubuntu-Bold.ttf', 96)

    lines = text.split('\n')

    line_max_length = lines[np.argmax([len(l) for l in lines])]
    LINE_W, LINE_H = font.getsize(line_max_length)

    fsize = (round(2.0*LINE_W), round(1.25*LINE_H*len(lines)))
    image = Image.new('L', fsize, color='black')

    draw = ImageDraw.Draw(image)
    # draw.multiline_text((0, 0), text, fill='white', font=font)

    char_bb = []
    space_w = font.getsize('O')[0]
    x, y = 0, 0
    for line in lines:
        x = 0  # carriage-return

        for ch in line:  # render each character
            if ch.isspace():  # just shift
                x += space_w
            else:
                # render the character
                draw.text((x, y), ch, fill='white', font=font)
                ch_size = font.getsize(ch)
                char_bb.append((x, y, ch_size[0], ch_size[1]))
                x += ch_size[0]

        y += LINE_H  # line-feed


    crop_box_x = min([box[0] for box in char_bb])
    crop_box_y = min([box[1] for box in char_bb])
    crop_box_w = max([box[0]+box[2] for box in char_bb])
    crop_box_h = max([box[1]+box[3] for box in char_bb])
    crop_box = (crop_box_x, crop_box_y, crop_box_w, crop_box_h)


    debug = image.convert('RGB')
    draw = ImageDraw.Draw(debug)
    for (x, y, w, h) in char_bb:
        draw.rectangle((x, y, x+w, y+h), outline=(255, 0, 0))
    draw.rectangle(crop_box, outline=(0, 255, 0))

    debug.show()

    words = ' '.join(text.split())
    print(words)
