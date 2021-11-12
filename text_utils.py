from __future__ import division
from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from PIL import Image, ImageDraw, ImageFont
import pickle
from pathlib import Path
import scipy.signal as ssig
import scipy.stats as sstat
import math


def sample_weighted(p_dict):
    ps = list(p_dict.keys())
    return p_dict[np.random.choice(ps, p=ps)]


def move_bb(bbs, t):
    """
    Translate the bounding-boxes in by t_x,t_y.
    BB : 2x4xn
    T  : 2-long np.array
    """
    return bbs + t[:, None, None]


def crop_safe(arr, rect, bbs=[], pad=0):
    """
    ARR : arr to crop
    RECT: (x,y,w,h) : area to crop to
    BBS : nx4 xywh format bounding-boxes
    PAD : percentage to pad

    Does safe cropping. Returns the cropped rectangle and
    the adjusted bounding-boxes
    """
    rect = np.array(rect)
    rect[:2] -= pad
    rect[2:] += 2*pad
    x1, y1 = max(0, rect[0]), max(0, rect[1])
    x2, y2 = [min(arr.shape[0], rect[0]+rect[2]),
              min(arr.shape[1], rect[1]+rect[3])]
    arr = arr[y1:y2, x1:x2]
    if len(bbs) > 0:
        for i in range(len(bbs)):
            bbs[i, 0] -= x1
            bbs[i, 1] -= y1
        return arr, bbs
    else:
        return arr


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


class RenderFont(object):
    """
    Outputs a rasterized font sample.
        Output is a binary mask matrix cropped closesly with the font.
        Also, outputs ground-truth bounding boxes and text string
    """

    def __init__(self, font_dir: Path, font_model_path: Path, text_path: Path):
        # distribution over the type of text:
        # whether to get a single word, paragraph or a line:
        self.p_text = {1.0: 'WORD',
                       0.0: 'LINE',
                       0.0: 'PARA'}

        # TEXT PLACEMENT PARAMETERS:
        self.f_shrink = 0.90
        self.max_shrink_trials = 5  # 0.9^5 ~= 0.6
        # the minimum number of characters that should fit in a mask
        # to define the maximum font height.
        self.min_nchar = 2
        self.min_font_h = 48  # px : 0.6*12 ~ 7px <= actual minimum height
        self.max_font_h = 320  # px
        self.p_flat = 0.10

        # curved baseline:
        self.p_curved = 1.0
        self.baselinestate = BaselineState.get_sample()
        # text-source : gets english text:
        self.text_source = TextSource(min_nchar=self.min_nchar,
                                      fn=text_path)

        # get font-state object:
        self.font_state = FontState(font_dir, font_model_path)

    def render_multiline(self, font: ImageFont.FreeTypeFont, text):
        """
        renders multiline TEXT on the pygame surface SURF with the
        font style FONT.
        A new line in text is denoted by \n, no other characters are
        escaped. Other forms of white-spaces should be converted to space.

        returns the updated surface, words and the character bounding boxes.
        """
        # get the number of lines
        lines = text.split('\n')

        line_max_length = lines[np.argmax([len(l) for l in lines])]
        LINE_W, LINE_H = font.getsize(line_max_length)

        fsize = (round(2.0*LINE_W), round(1.25*LINE_H*len(lines)))
        image = Image.new('L', fsize, color='black')
        draw = ImageDraw.Draw(image)

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

        # debug = image.convert('RGB')
        # draw = ImageDraw.Draw(debug)
        # for (x, y, w, h) in char_bb:
        #     draw.rectangle((x, y, x+w, y+h), outline=(255, 0, 0))
        # draw.rectangle(crop_box, outline=(0, 255, 0))

        # debug.show()

        words = ' '.join(text.split())

        image = np.array(image.crop(crop_box))
        char_bb = np.array(char_bb)
        return image, words, char_bb

    def render_curved(self, font: ImageFont.FreeTypeFont, word_text):  # add lang
        """
        use curved baseline for rendering word
        """
        def draw_char(font: ImageFont.FreeTypeFont, ch: str, rotation: float):
            ch_bb = list(font.getbbox(ch))
            ch_image = Image.new('RGBA', (ch_bb[2], ch_bb[3]), (0, 0, 0, 0))
            draw = ImageDraw.Draw(ch_image)
            draw.text((0, 0), ch, font=font, fill=(255, 255, 255, 255))
            ch_image = ch_image.rotate(rotation, Image.BICUBIC, expand=True)
            return ch_image

        wl = len(word_text)
        isword = len(word_text.split()) == 1

        if not isword or wl > 10 or np.random.rand() > self.p_curved:
            return self.render_multiline(font, word_text)

        word_bound = font.getbbox(word_text)
        fsize = (round(2.0*word_bound[2]), round(3*word_bound[3]))
        image = Image.new('L', fsize, color='black')

        # baseline state
        mid_idx = wl//2

        BS = BaselineState.get_sample()
        curve = [BS.curve(i-mid_idx) for i in range(wl)]
        curve[mid_idx] = -np.sum(curve) / (wl-1)
        rots = [math.degrees(math.atan(BS.differential(i-mid_idx)/(font.size/2)))
                for i in range(wl)]

        # pillow
        size = image.size
        ch_image = draw_char(font, word_text[mid_idx], rots[mid_idx])

        x = int((size[0] - ch_image.size[0]) / 2)
        y = int((size[1] - ch_image.size[1]) / 2 - curve[mid_idx])
        image.paste(ch_image, (x, y), mask=ch_image)
        mid_ch_bb = (x, y, ch_image.size[0], ch_image.size[1])

        char_bb = []
        last_bb = mid_ch_bb
        for i in range(wl):
            # skip the middle character
            if i == mid_idx:
                last_bb = mid_ch_bb
                char_bb.append(mid_ch_bb)
                continue
            elif i < mid_idx:  # left-chars
                i = mid_idx-1-i
            elif i > mid_idx:  # right-chars begin
                pass

            ch = word_text[i]

            # draw a single character to a separate image
            ch_bb = list(font.getbbox(ch))
            ch_image = draw_char(font, ch, rots[i])

            if i < mid_idx:
                x = last_bb[0] - ch_bb[2]
            elif i >= mid_idx:
                x = last_bb[0] + last_bb[2]
            y = int(last_bb[1] + 2 + curve[i])

            image.paste(ch_image, (x, y), mask=ch_image)
            ch_bb[0] = x
            ch_bb[1] = y
            last_bb = (x, y, ch_image.size[0], ch_image.size[1])
            char_bb.append(last_bb)

        crop_box_x = min([box[0] for box in char_bb])
        crop_box_y = min([box[1] for box in char_bb])
        crop_box_w = max([box[0]+box[2] for box in char_bb])
        crop_box_h = max([box[1]+box[3] for box in char_bb])
        crop_box = (crop_box_x, crop_box_y, crop_box_w, crop_box_h)

        # debug = image.convert('RGB')
        # draw = ImageDraw.Draw(debug)
        # for (x, y, w, h) in char_bb:
        #     draw.rectangle((x, y, x+w, y+h), outline=(255, 0, 0))
        # draw.rectangle(crop_box, outline=(0, 255, 0))
        # debug.show()
        # exit(0)

        word_image = np.array(image.crop(crop_box))
        char_bb = np.array(char_bb)

        # update box coordinates after cropping
        char_bb[:, 0] = char_bb[:, 0] - crop_box_x
        char_bb[:, 1] = char_bb[:, 1] - crop_box_y

        # plt.imshow(word_image)
        # plt.show()
        # exit()

        return word_image, word_text, char_bb

    def get_nline_nchar(self, mask_size, font_height, font_width):
        """
        Returns the maximum number of lines and characters which can fit
        in the MASK_SIZED image.
        """
        H, W = mask_size
        nline = int(np.ceil(H/(2*font_height)))
        nchar = int(np.floor(W/font_width))
        return nline, nchar

    def place_text(self, text_arrs: List[np.ndarray], back_arr, bbs: List[np.ndarray]):
        areas = [-np.prod(ta.shape) for ta in text_arrs]
        order = np.argsort(areas)

        locs = [None for i in range(len(text_arrs))]
        out_arr = np.zeros_like(back_arr)
        for i in order:
            ba = np.clip(back_arr.copy().astype(np.float), 0, 255)
            ta = np.clip(text_arrs[i].copy().astype(np.float), 0, 255)
            ba[ba > 127] = 1e8
            intersect = ssig.fftconvolve(ba, ta[:: -1, :: -1], mode='valid')
            safemask = intersect < 1e8

            if not np.any(safemask):  # no collision-free position:
                # warn("COLLISION!!!")
                return back_arr, locs[: i], bbs[: i], order[: i]

            minloc = np.transpose(np.nonzero(safemask))
            loc = minloc[np.random.choice(minloc.shape[0]), :]
            locs[i] = loc

            # update the bounding-boxes:
            bbs[i] = move_bb(bbs[i], loc[:: -1])

            # blit the text onto the canvas
            w, h = text_arrs[i].shape
            out_arr[loc[0]: loc[0]+w, loc[1]: loc[1]+h] += text_arrs[i]

        return out_arr, locs, bbs, order

    def robust_HW(self, mask):
        m = mask.copy()
        m = (~mask).astype('float')/255
        rH = np.median(np.sum(m, axis=0))
        rW = np.median(np.sum(m, axis=1))
        return rH, rW

    def sample_font_height_px(self, h_min, h_max):
        if np.random.rand() < self.p_flat:
            rnd = np.random.rand()
        else:
            rnd = np.random.beta(2.0, 2.0)

        h_range = h_max - h_min
        f_h = np.floor(h_min + h_range*rnd)
        return f_h

    def bb_xywh2coords(self, bbs):
        """
        Takes an nx4 bounding-box matrix specified in x,y,w,h
        format and outputs a 2x4xn bb-matrix, (4 vertices per bb).
        """
        n, _ = bbs.shape
        coords = np.zeros((2, 4, n))
        for i in range(n):
            coords[:, :, i] = bbs[i, : 2][:, None]
            coords[0, 1, i] += bbs[i, 2]
            coords[:, 2, i] += bbs[i, 2: 4]
            coords[1, 3, i] += bbs[i, 3]
        return coords

    def render_sample(self, font_name, font, mask):
        """
        Places text in the "collision-free" region as indicated
        in the mask -- 255 for unsafe, 0 for safe.
        The text is rendered using FONT, the text content is TEXT.
        """
        # H,W = mask.shape
        H, W = self.robust_HW(mask)

        # find the maximum height in pixels:
        max_font_h = min(0.9*H, W/(self.min_nchar+1))
        max_font_h = min(max_font_h, self.max_font_h)
        if max_font_h < self.min_font_h:  # not possible to place any text here
            return  # None

        # let's just place one text-instance for now
        # TODO : change this to allow multiple text instances?
        i = 0
        while i < self.max_shrink_trials and max_font_h > self.min_font_h:
            # if i > 0:
            #     print colorize(Color.BLUE, "shrinkage trial : %d"%i, True)

            # sample a random font-height:
            f_h_px = self.sample_font_height_px(self.min_font_h, max_font_h)
            # print "font-height : %.2f (min: %.2f, max: %.2f)"%(f_h_px, self.min_font_h,max_font_h)
            # convert from pixel-height to font-point-size:
            f_h = self.font_state.get_font_size(font_name, f_h_px)

            # update for the loop
            max_font_h = f_h_px
            i += 1

            # font.size = f_h  # set the font-size

            # compute the max-number of lines/chars-per-line:
            nline, nchar = self.get_nline_nchar(mask.shape[: 2], f_h, f_h)
            # print ('  > nline = {}, nchar = {}'.format(nline, nchar))

            if nchar < self.min_nchar:
                return None
            assert nline >= 1 and nchar >= self.min_nchar, f'nline={nline}, nchar={nchar}, min_nchar={self.min_nchar}'

            # sample text:
            text_type = sample_weighted(self.p_text)
            text = self.text_source.sample(nline, nchar, text_type)
            if len(text) == 0 or np.any([len(line) == 0 for line in text]):
                continue
            # print colorize(Color.GREEN, text)

            # render the text:
            txt_arr, txt, bb = self.render_curved(font, text)
            bb = self.bb_xywh2coords(bb)

            # debug = Image.fromarray(txt_arr).convert('RGB')
            # draw = ImageDraw.Draw(debug)
            # debug_boxes = bb.transpose()
            # for box in debug_boxes:
            #     draw.polygon(box.flatten().tolist(), outline=(255,0,0))

            # # for (x,y,w,h) in bb:
            # #     draw.rectangle([(x, y), (x+w, y+h)], outline=(255, 0, 0))

            # debug.show()
            # exit(0)

            # make sure that the text-array is not bigger than mask array:
            if np.any(np.r_[txt_arr.shape[:2]] > np.r_[mask.shape[:2]]):
                # warn("text-array is bigger than mask")
                continue

            # position the text within the mask:
            text_mask, loc, bb, _ = self.place_text([txt_arr], mask, [bb])
            if len(loc) > 0:  # successful in placing the text collision-free:
                return text_mask, loc[0], bb[0], text
        return  # None

    def visualize_bb(self, text_arr, bbs):
        ta = text_arr.copy()
        for r in bbs:
            cv2.rectangle(ta, (r[0], r[1]), (r[0]+r[2],
                          r[1]+r[3]), color=128, thickness=1)
        plt.imshow(ta, cmap='gray')
        plt.show()


class FontState(object):
    """
    Defines the random state of the font rendering
    """
    # size = [50, 10]  # normal dist mean, std
    size = [30, 70]  # normal dist mean, std
    underline = 0.05
    strong = 0.5
    oblique = 0.2
    wide = 0.5
    strength = [0.05, 0.1]  # uniform dist in this interval
    underline_adjustment = [1.0, 2.0]  # normal dist mean, std
    # beta distribution alpha, beta, offset, range (mean is a/(a+b))
    kerning = [2, 5, 0, 20]
    border = 0.25
    random_caps = -1  # don't recapitalize : retain the capitalization of the lexicon
    # lower case, upper case, proper noun
    capsmode = [str.lower, str.upper, str.capitalize]
    curved = 0.2
    random_kerning = 0.2
    random_kerning_amount = 0.1

    def __init__(self, font_dir: Path, font_model_path: Path, char_freq_path: Optional[Path] = None, create_model=False):

        # get character-frequencies in the English language:
        # with open(char_freq_path,'rb') as f:
        #   self.char_freq = cp.load(f)
        #    u = pickle._Unpickler(f)
        #   u.encoding = 'latin1'
        #   p = u.load()
        #   self.char_freq = p

        # get the model to convert from pixel to font pt size:
        with open(font_model_path, 'rb') as f:
            self.font_model = pickle.load(f)

        # get the names of fonts to use:
        self.fonts = sorted(font_dir.glob('**/*.ttf'))
        print(self.fonts)
        print(f'Total: {len(self.fonts)} font(s)')

    def get_aspect_ratio(self, font, size=None):
        """
        Returns the median aspect ratio of each character of the font.
        """
        if size is None:
            size = 12  # doesn't matter as we take the RATIO
        return 1.0
        # chars = ''
        # chars = ''.join(self.char_freq.keys())
        # w = np.array(self.char_freq.values())

        # get the [height,width] of each character:
        try:
            sizes = font.get_metrics(chars, size)
            good_idx = [i for i in range(len(sizes)) if sizes[i] is not None]
            sizes, w = [sizes[i] for i in good_idx], w[good_idx]
            sizes = np.array(sizes).astype('float')[:, [3, 4]]
            r = np.abs(sizes[:, 1]/sizes[:, 0])  # width/height
            good = np.isfinite(r)
            r = r[good]
            w = w[good]
            w /= np.sum(w)
            r_avg = np.sum(w*r)
            return r_avg
        except:
            return 1.0

    def get_font_size(self, font_name, font_size_px):
        """
        Returns the font-size which corresponds to FONT_SIZE_PX pixels font height.
        """
        m = self.font_model[font_name]
        return m[0]*font_size_px + m[1]  # linear model

    def sample(self):
        """
        Samples from the font state distribution
        """
        font = self.fonts[int(np.random.randint(0, len(self.fonts)))]
        font_name = font.stem
        return {
            'font': font,
            'name': font_name,
            'size': np.random.randint(self.size[0], self.size[1]),
            'underline': np.random.rand() < self.underline,
            'underline_adjustment': max(2.0, min(-2.0, self.underline_adjustment[1]*np.random.randn() + self.underline_adjustment[0])),
            'strong': np.random.rand() < self.strong,
            'oblique': np.random.rand() < self.oblique,
            'strength': (self.strength[1] - self.strength[0])*np.random.rand() + self.strength[0],
            'char_spacing': int(self.kerning[3]*(np.random.beta(self.kerning[0], self.kerning[1])) + self.kerning[2]),
            'border': np.random.rand() < self.border,
            'random_caps': np.random.rand() < self.random_caps,
            'capsmode': random.choice(self.capsmode),
            'curved': np.random.rand() < self.curved,
            'random_kerning': np.random.rand() < self.random_kerning,
            'random_kerning_amount': self.random_kerning_amount,
        }


class TextSource(object):
    """
    Provides text for words, paragraphs, sentences.
    """

    def __init__(self, min_nchar, fn):
        """
        TXT_FN : path to file containing text data.
        """
        self.min_nchar = min_nchar
        self.fdict = {'WORD': self.sample_word,
                      'LINE': self.sample_line,
                      'PARA': self.sample_para}

        with open(fn, 'r') as f:
            self.txt = [l.strip() for l in f.readlines()]
        # print(self.txt)

        # distribution over line/words for LINE/PARA:
        self.p_line_nline = np.array([0.85, 0.10, 0.05])
        self.p_line_nword = [4, 3, 12]  # normal: (mu, std)
        self.p_para_nline = [1.0, 1.0]  # [1.7,3.0] # beta: (a, b), max_nline
        self.p_para_nword = [1.7, 3.0, 10]  # beta: (a,b), max_nword

        # probability to center-align a paragraph:
        self.center_para = 0.5

    def check_symb_frac(self, txt, f=0.35):
        """
        T/F return : T iff fraction of symbol/special-charcters in
                     txt is less than or equal to f (default=0.25).
        """
        return np.sum([not ch.isalnum() for ch in txt])/(len(txt)+0.0) <= f

    def is_good(self, txt, f=0.35):
        """
        T/F return : T iff the lines in txt (a list of txt lines)
                     are "valid".
                     A given line l is valid iff:
                         1. It is not empty.
                         2. symbol_fraction > f
                         3. Has at-least self.min_nchar characters
                         4. Not all characters are i,x,0,O,-
        """
        def is_txt(l):
            char_ex = ['i', 'I', 'o', 'O', '0', '-']
            chs = [ch in char_ex for ch in l]
            return not np.all(chs)

        return [(len(l) > self.min_nchar
                 and self.check_symb_frac(l, f)
                 and is_txt(l)) for l in txt]

    def center_align(self, lines):
        """
        PADS lines with space to center align them
        lines : list of text-lines.
        """
        ls = [len(l) for l in lines]
        max_l = max(ls)
        for i in range(len(lines)):
            l = lines[i].strip()
            dl = max_l-ls[i]
            lspace = dl//2
            rspace = dl-lspace
            lines[i] = ' '*lspace+l+' '*rspace
        return lines

    def get_lines(self, nline, nword, nchar_max, f=0.35, niter=100):
        def h_lines(niter=100):
            lines = ['']
            iter = 0
            while not np.all(self.is_good(lines, f)) and iter < niter:
                iter += 1
                line_start = np.random.choice(len(self.txt)-nline)
                lines = [self.txt[line_start+i] for i in range(nline)]
            return lines

        lines = ['']
        iter = 0
        while not np.all(self.is_good(lines, f)) and iter < niter:
            iter += 1
            lines = h_lines(niter=100)
            # get words per line:
            nline = len(lines)
            for i in range(nline):
                words = lines[i].split()
                dw = len(words)-nword[i]
                if dw > 0:
                    first_word_index = random.choice(range(dw+1))
                    lines[i] = ' '.join(
                        words[first_word_index:first_word_index+nword[i]])

                # chop-off characters from end:
                while len(lines[i]) > nchar_max:
                    if not np.any([ch.isspace() for ch in lines[i]]):
                        lines[i] = ''
                    else:
                        lines[i] = lines[i][:len(
                            lines[i])-lines[i][::-1].find(' ')].strip()

        if not np.all(self.is_good(lines, f)):
            return  # None
        else:
            return lines

    def sample(self, nline_max, nchar_max, kind='WORD'):
        return self.fdict[kind](nline_max, nchar_max)

    def sample_word(self, nline_max, nchar_max, niter=100):
        rand_line = self.txt[np.random.choice(len(self.txt))]
        words = rand_line.split()
        if len(words) == 0:
            return []
        rand_word = random.choice(words)

        iter = 0
        while iter < niter and (not self.is_good([rand_word])[0] or len(rand_word) > nchar_max):
            rand_line = self.txt[np.random.choice(len(self.txt))]
            words = rand_line.split()
            if len(words) == 0:
                continue
            rand_word = random.choice(words)
            iter += 1

        if not self.is_good([rand_word])[0] or len(rand_word) > nchar_max:
            return []
        else:
            return rand_word

    def sample_line(self, nline_max, nchar_max):
        nline = nline_max+1
        while nline > nline_max:
            nline = np.random.choice([1, 2, 3], p=self.p_line_nline)

        # get number of words:
        nword = [self.p_line_nword[2]*sstat.beta.rvs(a=self.p_line_nword[0], b=self.p_line_nword[1])
                 for _ in range(nline)]
        nword = [max(1, int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            return '\n'.join(lines)
        else:
            return []

    def sample_para(self, nline_max, nchar_max):
        # get number of lines in the paragraph:
        nline = nline_max * \
            sstat.beta.rvs(a=self.p_para_nline[0], b=self.p_para_nline[1])
        nline = max(1, int(np.ceil(nline)))

        # get number of words:
        nword = [self.p_para_nword[2]*sstat.beta.rvs(a=self.p_para_nword[0], b=self.p_para_nword[1])
                 for _ in range(nline)]
        nword = [max(1, int(np.ceil(n))) for n in nword]

        lines = self.get_lines(nline, nword, nchar_max, f=0.35)
        if lines is not None:
            # center align the paragraph-text:
            if np.random.rand() < self.center_para:
                lines = self.center_align(lines)
            return '\n'.join(lines)
        else:
            return []
