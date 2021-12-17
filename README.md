# SynthText

```code
pip install "pillow>=8.4"
pip install gdown
```

```code
git clone https://github.com/VinhLoiIT/IndicSynthText.git
```

# prepare data

Download processed data
```
cd IndicSynthText
mkdir -p aic_data
cd aic_data

mkdir -p models
echo "Download colors_new.cp"
gdown https://drive.google.com/uc?id=1eEPexgSClRn0SPB7BIqs4r8F5mE68h1l -O models/colors_new.cp

echo "Download fonts"
gdown https://drive.google.com/uc?id=1bB4tO0AX9cxca0gIdHFIeWid6qLegiMm

echo "Download background"
gdown https://drive.google.com/uc?id=1-4xpbU8Yq4kRHIzBLw-c8j-FpbxDAskp

echo "Download depths"
gdown https://drive.google.com/uc?id=1HdMoDrTHu5wXs7KKihnxp2rfmcBCI7Rz 

echo "Download segs"
gdown https://drive.google.com/uc?id=1-2bOO-eREbtRqMdpj7oxnaGMKgOaIclL 

echo "Unzip bg.zip"
unzip bg.zip > /dev/null

echo "Unzip depths.zip"
unzip depths.zip > /dev/null

echo "Unzip segs.zip"
unzip segs.zip > /dev/null

echo "Unzip font.zip"
unzip font.zip > /dev/null

echo "Back to repo root dir"
cd ..
echo "Current dir:" $(pwd)
```

# Run

You might run this code once to get `font_px2pt.pkl`
```
python invert_font_size.py aic_data/vin-vnm.txt aic_data/Font --output_dir aic_data/models
```


```code
python gen_new.py ./aic_data \
--bg_dir ./aic_data/bg \
--depth_dir ./aic_data/depths \
--seg_dir ./aic_data/segs \
--text_path aic_data/vin-vnm.txt \
--output_dir icdar_outputs \
--viz \
--font_dir aic_data/Font
```