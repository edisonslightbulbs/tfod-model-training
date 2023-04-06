# A minimal template for training an object detection model

[![GitHub stars](https://img.shields.io/github/stars/edisonslightbulbs/tfod-model-training)](https://github.com/edisonslightbulbs/tfod-model-training/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/edisonslightbulbs/tfod-model-training)](https://github.com/edisonslightbulbs/tfod-model-training/network)
[![GitHub license](https://img.shields.io/github/license/edisonslightbulbs/tfod-model-training.svg?style=flat-square)](https://github.com/edisonslightbulbs/tfod-model-training/blob/main/LICENSE)

This project demonstrates transfer learning using Tensorflow.
The underlying concepts provide a solid foundation for training machine learning models.

### Quick Start: Setup the development environment

1. Install [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html) or [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html).

2. Create a virtual environment with Python 3.7.3, no default packages, and activate it.

    ```zsh
    conda create --name myenv --no-default-packages python=3.7
    conda activate myenv
    ```

3. Install the required packages.

    ```zsh
    # from the repository's root directory:
    pip install -r requirements.txt
    ```

4. Install `ipykernel`:

    ```zsh
    python -m ipykernel install --user --name=myenv
    ```

### Quick Start: Collect and label images

1. Collect target images in `./data/raw/`.

2. Label images using `labelImg`.

    a. From terminal run:

    ```zsh
    labelImg
    ```

    b. Using `labelImg`, navigate to `./data/raw/` and label images.

## Running Project: Using CLI 

1. Setup:

    `python cli.py --setup --image-format < > --pretrained-model-url < > --pretrained-model-name < >`

    E.g.
    ```zsh
    python cli.py --setup --image-format jpg --pretrained-model-url http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz --pretrained-model-name ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8
    ```

2. Train:

    `python cli.py --train --batch-size < > --epochs < > --train-test-split < > --pretrained-model-ckpt < >`

    E.g.
    ```zsh
    python cli.py --train --batch-size 4 --epochs 1000 --train-test-split 0.7 --pretrained-model-ckpt 0
    ```

3. Smoke test:

    `python cli.py --validate --trained-model-ckpt < >`

    E.g.
    ```zsh
    python cli.py --validate --trained-model-ckpt 1
    ```
### Running Project: Using notebooks

From the repository's root directory, start a Jupyter Notebook session and select the kernel initialized during setup.

1. Step through `./notebooks/setup.ipynb` to set up the project.
   
2. Step through `./notebooks/train.ipynb` to train an object detection model.
 
3. Step through `./notebooks/validate.ipynb` to smoke test the trained image-object detection model.


---

###### N.b., to enable GPU-base training, check your TensorFlow version and ensure [tallying CUDA and CUDNN versions](https://www.tensorflow.org/install/source_windows) are installed.


# Promotion

👏 [Star if you liked](https://github.com/edisonslightbulbs/tfod-model-training/stargazers) 

👐 [Share if you loved](https://github.com/edisonslightbulbs/tfod-model-training "Copy project link")

🎓 [Fork to contribute](https://github.com/edisonslightbulbs/tfod-model-training/fork)
