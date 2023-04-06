## # Deploy

import os
import git

repo = git.Repo(".", search_parent_directories=True)
rootdir = repo.git.rev_parse("--show-toplevel")


## #### Setup work directories
dirs = {'export': os.path.join(rootdir, 'models', 'export'),
        'deploy': os.path.join(rootdir, 'models', 'deploy'),
        'trained model': os.path.join(rootdir, 'models', 'trained'),
        'tfjs model': os.path.join(rootdir, 'models', 'deploy', 'tfjs'),
        'tflite model': os.path.join(rootdir, 'models', 'deploy', 'tflite'),
        'tensorflow': os.path.join(rootdir, 'external','Tensorflow', 'models')}

for k,v in dirs.items():
    print(f'{k} directory: {v}')
    if not os.path.exists(v):
        os.makedirs(v, exist_ok=True)


## #### Freeze graph
tensor = 'image_tensor'
exportdir = os.path.join(dirs['export'])
trained_modeldir = os.path.join(dirs['trained model'])
pipeconfig = os.path.join(dirs['trained model'], 'pipeline.config')
script = os.path.join(dirs['tensorflow'], 'research', 'object_detection', 'exporter_main_v2.py')

cmd = f'python {script} --input_type={tensor} --pipeline_config_path={pipeconfig} --trained_checkpoint_dir={trained_modeldir} --output_directory={exportdir}'
os.system(cmd)


## #### Export TFJS format
input_format = 'tf_saved_model'
output_format = 'tfjs_graph_model'
tfjs_modeldir = os.path.join(dirs['tfjs model'])

nodes = ['detection_boxes',
         'detection_classes',
         'detection_features',
         'detection_multiclass_scores',
         'detection_scores',
         'num_detections',
         'raw_detection_boxes',
         'raw_detection_scores']

output_nodes = ','.join(nodes)
frozen_modeldir = os.path.join(exportdir, 'saved_model')
name = f'serving_default {frozen_modeldir} {tfjs_modeldir}'

exe = 'tensorflowjs_converter'
cmd = f'{exe} --input_format={input_format} --output_node_names={output_nodes} --output_format={output_format} --signature_name={name}'

os.system(cmd)


## #### Export TFLite format
tflite_modeldir = os.path.join(dirs['tflite model'])
tflite_model = os.path.join(tflite_modeldir, 'detect.tflite')
script = os.path.join(dirs['tensorflow'], 'research', 'object_detection', 'export_tflite_graph_tf2.py ')

cmd = f'python {script} --pipeline_config_path={pipeconfig} --trained_checkpoint_dir={trained_modeldir} --output_directory={tflite_modeldir}'
os.system(cmd)

shapes = '1,300,300,3'
arrays = ['TFLite_Detection_PostProcess',
          'TFLite_Detection_PostProcess:1',
          'TFLite_Detection_PostProcess:2',
          'TFLite_Detection_PostProcess:3']

output_arrays = ','.join(arrays)
input_arrays = 'normalized_input_image_tensor'
params = '--inference_type=FLOAT --allow_custom_ops'
frozen_modeldir = os.path.join(tflite_modeldir, 'saved_model')

exe = 'tflite_convert'
cmd = f'{exe} --saved_model_dir={frozen_modeldir} --output_file={tflite_model} --input_shapes={shapes} --input_arrays={input_arrays} --output_arrays={output_arrays} {params}'
os.system(cmd)
