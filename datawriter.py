from typing import Dict, List
import io
import lmdb
from PIL import Image
from pathlib import Path
import warnings
import json


class DataWriter:
    pass


class LMDBDataWriter(DataWriter):
    def __init__(self, out_path, max_samples, map_size=1099511627776):
        self.out_path = out_path
        self.map_size = map_size
        self.max_samples = max_samples
        self.cache_size = 1000

    def open(self):
        self._cache = {}
        self._counter = 0
        self.env = lmdb.open(self.out_path, map_size=self.map_size)

    def write(self, results: List[Dict]):
        imageKey = 'image-%09d' % self._counter
        labelKey = 'label-%09d' % self._counter

        for result_idx, result in enumerate(results):
            img = result['img']
            nw = len(result['txt'])
            print('number of words', nw)
            label = result['txt']
            print(label)
            # if filter_text(lang, label):
            #     print('invalid word encountered')
            #     print(label)
            #     continue
            bbox = result['wordBB']   # (2, 4, num_words)
            print('bbox shape', bbox.shape)
            try:
                image_pil = Image.fromarray(img)
            except ValueError:
                continue
            imgByteArr = io.BytesIO()
            image_pil.save(imgByteArr, format='PNG')
            imgByteArr = imgByteArr.getvalue()

            self._cache[imageKey] = imgByteArr
            self._cache[labelKey] = label

            if (self._counter + 1) % self.cache_size == 0:
                self._write_cache()
                print('Written %d samples' % (self._counter))

            if self._counter == self.max_samples:
                self.close()
                break

        self._write_cache()

    def _write_cache(self):
        with self.env.begin(write=True) as txn:
            for k, v in self._cache.items():
                txn.put(k, v)
        self._cache = {}

    def close(self):
        # cache['num-samples'.encode()] = str(cnt-1).encode()
        # writeCache(env, cache)
        self.env.close()


class FolderWriter(DataWriter):

    def __init__(self,
                 output_dir: str,
                 max_samples: int,
                 char_mask: bool = False,
                 word_mask: bool = False,
                 char_box: bool = False,
                 word_box: bool = False,
                 image_ext: str = '.png'):
        self.output_dir = Path(output_dir)
        self.max_samples = max_samples
        self.image_ext = image_ext

        self.image_out_dir = self.output_dir / "images"
        self.image_out_dir.mkdir(exist_ok=True, parents=True)
        self.label_dir = self.output_dir / "labels"
        self.label_dir.mkdir(exist_ok=True, parents=True)

        if any([char_mask, word_mask, char_box, word_box]):
            warnings.warn(
                "char_mask, word_mask, char_box, word_box will be available in the future")

    def open(self):
        self._counter = 0

    def write(self, results: List[Dict]):

        for result_idx, result in enumerate(results):
            img = result['img']
            nw = len(result['txt'])
            label = result['txt']
            bbox = result['wordBB']   # (2, 4, num_words)
            bbox = bbox.transpose()  # (num_words, 4, 2)
            try:
                image_pil = Image.fromarray(img)
            except ValueError:
                continue

            image_name = f'image-{self._counter:09d}{self.image_ext}'
            image_path = self.image_out_dir / image_name
            image_pil.save(image_path, format='PNG')

            label_name = f'label-{self._counter:09d}.json'
            label_path = self.label_dir / label_name

            with open(label_path, 'wt', encoding='utf-8') as f:
                json_dict = {
                    'version': 1.0,
                    'image_name': image_name,
                    'word_boxes': []
                }
                for i, word in enumerate(result['txt']):
                    item = {
                        'text': word,
                        'bbox': bbox[i].tolist()
                    }
                    json_dict['word_boxes'].append(item)

                json.dump(json_dict, f, indent=2)

            self._counter += 1

            if self._counter == self.max_samples:
                break

    def close(self):
        pass
