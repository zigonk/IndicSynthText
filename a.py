from PIL import Image, ImageFont, ImageDraw
import numpy as np
import math


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

    # word_text = 'aBg'
    word_text = 'g'
    wl = len(word_text)
    isword = len(word_text.split()) == 1
    font = ImageFont.truetype(
        'SynthTextGen/fonts/ubuntuen/Ubuntu-Bold.ttf', 200)

    ascent, descent = font.getmetrics()
    print('metrics', ascent, descent)

    w, h = font.getsize(word_text)
    print('size', w, h)

    offsetx, offsety = font.getoffset(word_text)
    print('offset', offsetx, offsety)

    image = draw_char(font, word_text, 0)
    print('Image Size', image.size)
    draw = ImageDraw.Draw(image)

    baseline = image.size[1] - descent
    print('baseline', baseline)
    # draw ascent
    draw.rectangle((offsetx, offsety, offsetx+w, ascent), fill=(0,0,255,127))
    # draw descent
    draw.rectangle((offsetx, baseline, offsetx+w, baseline+abs(descent)), fill=(255,0,0,127))

    image.show()
