# SynthText
Refactored code for generating Indic synthetic text images as described in ["Synthetic Data for Text Localisation in Natural Images", Ankush Gupta, Andrea Vedaldi, Andrew Zisserman, CVPR 2016](http://www.robots.ox.ac.uk/~vgg/data/scenetext/).


**Synthetic Scene-Text Image Samples**
![Synthetic Scene-Text Samples](medium_art.png)

The code in the `master` branch is for Python2. Python3 is supported in the `python3` branch.

The main dependencies are:

```
pygame, opencv (cv2), PIL (Image), numpy, matplotlib, h5py, scipy, kaggle
```
### Downloading Data needed for generation
In the project directory run
```
kaggle datasets download azharshaikh/SynthTextGen
```
This will download a zip file named `SynthTextGen`. Ensure you have unzipped file the to ./SynthTextGen. This data file includes:

  - **dset.h5**: This is a sample h5 file which contains a set of 5 images along with their depth and segmentation information. Note, this is just given as an example; you are encouraged to add more images (along with their depth and segmentation information) to this database for your own use.
  - **data/fonts**: Sample fonts for the indic languages(add more fonts to this folder and then update `fonts/fontlist.txt` with their paths).
  - **data/newsgroup**: Text cropus for the indic languages used to render scene text. Look inside `text_utils.py` to see how the text inside this file is used by the renderer.
  - **data/models/colors_new.cp**: Color-model (foreground/background text color model), learnt from the IIIT-5K word dataset.
  - **data/models**: Other cPickle files (**char\_freq.cp**: frequency of each character in the text dataset; **font\_px2pt.cp**: conversion from pt to px for various fonts: If you add a new font, make sure that the corresponding model is present in this file, if not you can add it by adapting `invert_font_size.py`).

### Generating samples
After downloading data run 

```
python gen.py --viz --output_path path/to/store/generated_images --total_samples --lang
```
This script will generate scene-text image samples and store them in an lmdb file at the output path specified. If the `--viz` option is specified, the generated output will be visualized as the script is being run; omit the `--viz` option to turn-off the visualizations. If you want to visualize later, run:

```
python visualize_results.py
```
### Adding a new language
- Download the language font (.ttf file) to data/fonts/ (ex data/fonts/hin.ttf) then update fonts/fontlist.txt with the font's path.
- Create a new language font model (.cp file) using invert_font_size.py and the laguage font and place it in data/models/
- Add a language text file (like wikipedia articles, news etc) to data/newsgroup with the name newsgrouplangname(ex newsgrouphin)
- *make sure the same lang name is used for both the font and text files.*
- To start generating samples refer to the *Generating samples* section.

Additional Instructions can be found [here](https://github.com/ankush-me/SynthText)


