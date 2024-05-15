# fsdl-glasses

In this repository, I build a neural network which takes in a picture of a person and returns whether that person is wearing glasses or not.
It is based on an old project I did [Glasses-Nerual-Network](https://github.com/shahzebbb/Glasses-Neural-Network) except this time it is a proper machine learning code-base with multiple functionalities for training which I learnt from the [Full Stack Deep Learning Course](https://fullstackdeeplearning.com/course/2022/).

The code-base makes heavy use of Pytorch and Pytorch-Lightning.

## Setup

Using conda, create an environment using:

```py
conda create --name myenv python=3.10 pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia
```
and then install the packages in `requirements.txt` using:
```
pip install -r requirements.txt
```

In the future, I plan to setup an environment.yaml file to make setup easier similar to what is done in the [Full Stack Deep Learning Course Repo](https://github.com/the-full-stack/fsdl-text-recognizer-2022-labs/tree/main/setup)

Finally run the following path to set the Python Paths:

```py
echo "export PYTHONPATH=.:$PYTHONPATH" >> ~/.bashrc
```

## Description of code-base

### Dataset

Under `glasses_detector/data' you will find the code for all the data modules used to download and setup our data. Specifically, `glasses.py` contains the data module for the glasses detector model.

Under `glasses_detector/data/utils.py` you will find code to set up a PyTorch Dataset class, split the dataset and download the data from a URL.

### Models

Under `glasses_detector/models` you will find some CNN models you can train in the code-base.
Under 'glasses_detector/lit_models` you will find a `LightningModule` class which specifies training and validation steps, along with how to log metrics and setup optimizers.

### Metadata

Under `glasses_detector/metadata` you will find metadata for the project such as URLS, image sizes, URLs, etc.

### Credits

You can find the original dataset in this [repo.](https://github.com/cleardusk/MeGlass)
I also want to credit the [Full Stack Deep Learning Course](https://fullstackdeeplearning.com/course/2022/) as I used most of the code from there as a guideline to how I should structure this repository.
