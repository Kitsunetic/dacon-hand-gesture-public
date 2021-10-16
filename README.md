This is 3rd place solution of 
[**Ego-Vision Hand Gesture Recognition AI Contest**](https://dacon.io/en/competitions/official/235805/overview/description).

## Requirements

- Albumentations
- opencv-python
- imageio
- numpy
- pandas
- timm
- torch==1.7.0 with cuda toolkit 11.2.2, cudnn8
- pyaml
- adabelief_pytorch
- scikit-learn
- tqdm

## Training from the Scratch

1. Before running training, you have to download the original dataset from 
[dacon](https://dacon.io/en/competitions/official/235805/data).
2. After that, please follow 
[`dataset-gen.ipynb`](./dataset-gen.ipynb) 
notebook to generate new dataset.
3. To start training, please run

```bash
python main.py
```

## Inference Only

1. Download the original dataset from 
[dacon](https://dacon.io/en/competitions/official/235805/data).
2. Download the whole pretrained weights by running command bellow.  
If your environment not support `wget` command, please download pretrained weights from [here](https://github.com/Kitsunetic/dacon-hand-gesture-public/releases/tag/weights) manually, and locate them into `./results` directory.
```bash
wget -i "https://raw.githubusercontent.com/Kitsunetic/dacon-hand-gesture-public/master/pretrained_weights.txt" -P results
``` 
3. Follow [`inference.ipynb`](./inference.ipynb) notebook.
