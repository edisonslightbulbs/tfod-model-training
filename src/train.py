## # Train model

import os
import git
import json
import shutil

repo = git.Repo(".", search_parent_directories=True)
rootdir = repo.git.rev_parse("--show-toplevel")

## #### Load project parameters
path = os.path.join(rootdir, "config.json")
with open(path, "r") as file:
    parameters = json.load(file)

## #### Setup work directories
dirs = {'data': os.path.join(rootdir, 'data'),
        'labels': os.path.join(rootdir, 'data', 'labels'),
        'records': os.path.join(rootdir, 'data', 'records'),
        'trained model': os.path.join(rootdir, 'models', 'trained'),
        'pretrained model': os.path.join(rootdir, 'models', 'pre-trained'),
        'tensorflow': os.path.join(rootdir, 'external','Tensorflow', 'models')}

for k,v in dirs.items():
    print(f'{k} directory: {v}')
    if not os.path.exists(v):
        os.makedirs(v, exist_ok=True)


## #### Create label map and tensorflow records
#
# prepare train and test data for an object detection model by converting images
# and labels to TFRecord format, which is a binary file format for training
# tensorflow models
#
print("generating label map ...")
script = os.path.join(rootdir, 'src',  'labels.py')
cmd = f'python {script}'
labelmap = os.path.join(dirs['labels'],  'map.pbtxt')
os.system(cmd)

print("splitting test and train data...")
script = os.path.join(rootdir, 'src',  'partition.py')
cmd = f'python {script}'
os.system(cmd)

script = os.path.join(dirs['records'],  'generate.py')

print("generating train record ...")
images = os.path.join(dirs['data'], 'train')
record = os.path.join(dirs['records'], 'train.record')
cmd = f'python {script} -x {images} -l {labelmap} -o {record}'
os.system(cmd)

print("generating test record ...")
images = os.path.join(dirs['data'], 'test')
record = os.path.join(dirs['records'], 'test.record')
cmd = f'python {script} -x {images} -l {labelmap} -o {record}'
os.system(cmd)


## #### Copy pipeline configuration of pretrained model
pretrained_model = parameters['pretrained model']['name']
src = os.path.join(dirs['pretrained model'], pretrained_model, 'pipeline.config')
dst = os.path.join(dirs['trained model'])

try:
    shutil.copy(src, dst)
    os.chmod(dst, 0o666)
except FileExistsError:
    pass


## #### Setup pipeline configuration for model training
import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2

pipeconfig = os.path.join(dirs['trained model'], 'pipeline.config')
assert os.path.exists(pipeconfig), f"Unable to resolve: {pipeconfig}"

pipeline = pipeline_pb2.TrainEvalPipelineConfig()

with tf.io.gfile.GFile(pipeconfig) as file:
    proto_str = file.read()
    text_format.Merge(proto_str, pipeline)

pipeline.model.ssd.num_classes = len(labelmap)
test_record = os.path.join(dirs['records'], 'test.record')
train_record = os.path.join(dirs['records'], 'train.record')
ckpt = parameters['pretrained model']['ckpt']
ckpt_path = os.path.join(dirs['pretrained model'], pretrained_model, 'checkpoint', f'ckpt-{ckpt}')

pipeline.train_config.batch_size = parameters['batch size']
pipeline.train_config.fine_tune_checkpoint = ckpt_path
pipeline.train_config.fine_tune_checkpoint_type = "detection"
pipeline.train_input_reader.label_map_path= labelmap
pipeline.train_input_reader.tf_record_input_reader.input_path[:] = [train_record]

pipeline.eval_input_reader[0].label_map_path = labelmap
pipeline.eval_input_reader[0].tf_record_input_reader.input_path[:] = [test_record]


## #### Check pipeline config
# 
# (optional)
# 
# print(text_format.MessageToString(pipeline))


## #### Write pipeline config
with tf.io.gfile.GFile(pipeconfig, 'wb') as file:
    file.write(text_format.MessageToString(pipeline))


## #### Train model
epochs = parameters['epochs']
modeldir = dirs['trained model']
script = os.path.join(dirs['tensorflow'], 'research', 'object_detection', 'model_main_tf2.py')

cmd = f'python {script} --model_dir={modeldir} --pipeline_config_path={pipeconfig} --num_train_steps={epochs}'
os.system(cmd)

cmd = f'python {script} --model_dir={modeldir} --pipeline_config_path={pipeconfig} --checkpoint_dir={modeldir}'
os.system(cmd)
