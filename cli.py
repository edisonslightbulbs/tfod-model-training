import os
import git
import sys
import json
import argparse
from traceback import format_exc
from textwrap import dedent, indent

from utils import notebook

repo = git.Repo(".", search_parent_directories=True)
rootdir = repo.git.rev_parse("--show-toplevel")


def update_config(params: dict):
    path = os.path.join(rootdir, "config.json")
    try:
        with open(path, "r") as file:
            config = json.load(file)
    except (json.JSONDecodeError, FileNotFoundError):
        config = {}

    def update_nested_dict(base: dict, update: dict):
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                update_nested_dict(base[key], value)
            else:
                base[key] = value

    update_nested_dict(config, params)
    with open(path, "w") as file:
        json.dump(config, file, indent=4)


def deploy():
    try:
        script = os.path.join(rootdir, 'src', 'deploy.py')
        cmd = f'python {script}'
        os.system(cmd)

    except BaseException:
        print(format_exc())


def validate(ckpt:int):
    params = {'trained model': { 'ckpt': ckpt}}
    update_config(params)

    try:
        script = os.path.join(rootdir, 'src', 'validate.py')
        cmd = f'python {script}'
        os.system(cmd)

    except BaseException:
        print(format_exc())


def train(split=0.7 , batchsize=4, epochs=1000, ckpt=0):
    params = {'split': split,
              'epochs': epochs,
              'batch size': batchsize,
              'pretrained model': { 'ckpt': ckpt}}

    update_config(params)

    try:
        script = os.path.join(rootdir, 'src', 'train.py')
        cmd = f'python {script}'
        os.system(cmd)

    except BaseException:
        print(format_exc())


def setup(model_url: str, model_name: str, image_format='jpg'):
    params = {'image format': image_format,
              'pretrained model': {'url': model_url,
                             'name': model_name}}
    update_config(params)
    try:
        script = os.path.join(rootdir, 'src', 'setup.py')
        cmd = f'python {script}'
        os.system(cmd)

    except BaseException:
        print(format_exc())


def ipynb(script):
    try:
        script_name = os.path.basename(script)
        filename, _ = os.path.splitext(script_name)

        notebook_name = f'{filename}.ipynb'
        notebook_path = os.path.join(rootdir, 'notebooks', notebook_name)
        notebook.parse(script=script, notebook=notebook_path)

    except BaseException:
        print(format_exc())

def format_text(text:str, indentation: int)->str:
    return indent(dedent(text).strip(), ' ' * indentation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI: Transfer learning using Tensorflow.",
                                     formatter_class=argparse.RawTextHelpFormatter)

    help = """Specifies the url of the pretrained model."""
    help = format_text(text=help, indentation=6)
    parser.add_argument('--pretrained-model-url',
                        help=help,
                        type=str,
                        default='',
                        metavar='STR',
                        required="--setup" in sys.argv)

    help = """Specifies the name of the pretrained model."""
    help = format_text(text=help, indentation=6)
    parser.add_argument('--pretrained-model-name',
                        help=help,
                        type=str,
                        default='',
                        metavar='STR',
                        required="--setup" in sys.argv)

    help = """Specifies a pretrained model checkpoint for fine-tuning."""
    help = format_text(text=help, indentation=6)
    parser.add_argument('--pretrained-model-ckpt',
                        help=help,
                        type=int,
                        default=0,
                        metavar='INT',
                        required="--train" in sys.argv)

    help = """Specifies image format, e.g., jpg, jpeg, png, etc."""
    help = format_text(text=help, indentation=6)
    parser.add_argument('--image-format',
                        help=help,
                        type=str,
                        default='jpg',
                        metavar='STR',
                        required="--setup" in sys.argv)

    help = """Specifies the ratio for splitting the data into training and a
              test data. """
    help = format_text(text=help, indentation=6)
    parser.add_argument('--train-test-split',
                        help=help,
                        nargs='?',
                        type=float,
                        default=0.7,
                        metavar='FLOAT')

    help = """Specifies the number of training examples used in each iteration
              of the optimization algorithm during training. Specifically, the
              batch size determines how many examples the algorithm processes at
              once before updating the model's parameters."""

    help = format_text(text=help, indentation=6)
    parser.add_argument('--batch-size',
                        help=help,
                        type=int,
                        nargs='?',
                        default=4,
                        metavar='INT')

    help = """Specifies the number of times the model will be trained."""
    help = format_text(text=help, indentation=6)
    parser.add_argument('--epochs',
                        help=help,
                        type=int,
                        nargs='?',
                        default=1000,
                        metavar='INT')

    help = """Deploys trained model."""
    help = format_text(text=help, indentation=6)
    parser.add_argument('--deploy',
                        help=help,
                        default=False,
                        action='store_true')

    help = """Specifies a trained model checkpoint for validation."""
    help = format_text(text=help, indentation=6)
    parser.add_argument('--trained-model-ckpt',
                        help=help,
                        type=int,
                        default=1,
                        metavar='INT',
                        required="--validate" in sys.argv)

    example = f'Example: python cli.py --validate --trained-model-ckpt {2}'
    help = f"""
               Executes a smoke test on a trained model checkpoint.
               If this flag is set, the following arguments are required:
               --trained-model-ckpt

               {example}"""

    help = format_text(text=help, indentation=6)
    parser.add_argument('--validate', '-v',
                        help=help,
                        default=False,
                        action='store_true')

    bs = 4
    ckpt = 3
    split = 0.7
    epochs = 1000
    example = f'Example: python cli.py --train --batch-size {bs} --epochs {epochs} --train-test-split {split} --pretrained-model-ckpt {ckpt}'
    help = f"""
               Sequences model training based on a pretrained-model checkpoint.
               If this flag is set, the following arguments are required:
               --batch-size, --epochs, --train-test-split, --pretrained-model-ckpt

               {example}"""

    help = format_text(text=help, indentation=6)
    parser.add_argument('--train', '-t',
                        help=help,
                        default=False,
                        action='store_true')

    model_name = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
    model_url = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
    example = f'Example: python cli.py --setup --image-format jpg --pretrained-model-url {model_url} --pretrained-model-name {model_name}'

    help = f"""Specifies whether to initialize project dependencies. If this flag
               is set to True, the program will initialize required dependencies
               before running. If False or not set, the program will assume that
               dependencies are already installed and will not attempt to install
               them. Default value is False. If this flag is set, the following
               arguments are required:
               --image-format, --pretrained-model-url, --pretrained-model-name

               {example}"""

    help = format_text(text=help, indentation=6)
    parser.add_argument('--setup', '-s',
                        help=help,
                        default=False,
                        action='store_true')

    help = """Parses a target python script into a jupyter notebook in ./notebooks/.
              N.b.: use relative path to point to a target script. """

    help = format_text(text=help, indentation=6)
    parser.add_argument('--nbformat',
                        type=str,
                        default='',
                        help=help,
                        required=False)

    args = parser.parse_args()

    if args.setup:
        print('-- setting up model training environment')
        format = args.image_format
        model_url = args.pretrained_model_url
        model_name = args.pretrained_model_name
        setup(model_name=model_name, model_url=model_url, image_format=format)

    if args.train:
        print('-- training model')
        epochs = args.epochs
        batchsize = args.batch_size
        split = args.train_test_split
        ckpt = args.pretrained_model_ckpt
        train(split=split, batchsize=batchsize, epochs=epochs, ckpt=ckpt)

    if args.validate:
        print('-- smoke testing model')
        ckpt = args.trained_model_ckpt
        validate(ckpt=ckpt)

    if args.deploy:
        print('-- deploys trained model')
        deploy()

    #
    # show parameters
    #
    with open('config.json') as f:
        data = json.load(f)

    print('-- parameters:')
    info = json.dumps(data, indent=4)
    print(indent(info, ' ' * 3))

    if args.nbformat != '':
        script = os.path.join(rootdir, args.nbformat)

        if os.path.exists(script):
            print(f'-- parsing {script} to jupyter notebook')
            ipynb(script=script)

        else:
            print('-- unable to resolve script path')
