## # Setup 

import os
import sys
import git
import json
import shutil
import zipfile
import tarfile
import requests

repo = git.Repo(".", search_parent_directories=True)
rootdir = repo.git.rev_parse("--show-toplevel")

## #### Load project parameters
path = os.path.join(rootdir, "config.json")
with open(path, "r") as file:
    parameters = json.load(file)

## #### Check environment 
os.system('conda env list')
print(f'work directory: {rootdir}')
print(f'python environment: {sys.version}')


## #### Setup work directories
dirs = {'protoc': os.path.join(rootdir, 'res', 'protoc'),
        'pretrained model': os.path.join(rootdir, 'models', 'pre-trained'),
        'tensorflow': os.path.join(rootdir, 'external','Tensorflow', 'models')}

for k,v in dirs.items():
    print(f'{k} directory: {v}')
    if not os.path.exists(v):
        os.makedirs(v, exist_ok=True)


## #### Clone tensorflow models
directory = dirs['tensorflow']
if not os.path.exists(os.path.join(directory, 'research', 'object_detection')):
    repository = 'https://github.com/tensorflow/models'
    print(f'cloning {repository} ...')
    os.system(f'git clone {repository} {directory}')


## download-and-extract helper 
def download(url, filename):
    with requests.get(url) as response:
        ext = os.path.splitext(filename)[1]
    
        with open(filename, 'wb') as file:
            file.write(response.content)
    
        if ext == '.zip':
            with zipfile.ZipFile(filename, 'r') as zip:
                zip.extractall()
    
        if ext == '.gz':
            with tarfile.open(filename, 'r') as tar:
                tar.extractall()


## #### Setup tensorflow
directory = dirs['protoc']
os.chdir(directory)

print('downloading protoc.exe ...')
url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
filename = 'protoc-3.15.6-win64.zip'
download(url, filename)

print('appending protoc.exe to environment PATH ...')
os.environ['PATH'] += os.pathsep + os.path.abspath(os.path.join(os.getcwd(), 'bin'))

directory = os.path.join(dirs['tensorflow'], 'research')
os.chdir(directory)

print('compiling and installing protocol buffer files ...')
os.system('protoc object_detection/protos/*.proto --python_out=.')
os.system('pip install .')

print('building and installing tensorflow ...')
dst =  os.path.join(directory, 'setup.py')
src = os.path.join(directory, 'object_detection', 'packages', 'tf2', 'setup.py')
shutil.copy(src, dst)

os.system('python setup.py build')
os.system('python setup.py install')

print('installing tensorflow\'s slim library ...')
os.chdir(os.path.join(directory,  'slim'))
os.system('pip install -e .')

os.chdir(os.path.join(directory, 'object_detection', 'builders'))

print('testing object detection model builder ...')
script = os.path.join(directory, 'object_detection', 'builders', 'model_builder_tf2_test.py')
os.system(f'python {script}')


## #### Verify CUDA & CUDNN versions
import torch
import tensorflow as tf

msg = 'check https://www.tensorflow.org/install/source_windows to ensure tallying CUDA and CUDNN versions are installed'
print(msg)
print(f'tensorflow: {tf.__version__}')
print(f'cuda: {torch.version.cuda}')
print(f'cudnn: {torch.backends.cudnn.version()}')

## #### Smoke test object detection module
import object_detection


## #### Download pre-trained model
directory = os.path.join(rootdir, 'res')
os.chdir(directory)
pretrained_model_name = parameters['pretrained model']['name']
pretrained_model_url = parameters['pretrained model']['url']

# @todo: clever way of determining the file extension from url
download(url=pretrained_model_url, filename=f'{pretrained_model_name}.tar.gz')

src = pretrained_model_name
dst = os.path.join(dirs['pretrained model'], pretrained_model_name)

try:
    shutil.copytree(src, dst)
except FileExistsError:
    pass
