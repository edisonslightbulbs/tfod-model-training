## # Smoke test

import os
import git
import cv2
import json
import glob
import numpy as np

import tensorflow as tf
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.builders import model_builder
from object_detection.utils import visualization_utils as viz_utils

#
# import pyplot and sets up the notebook 
# environment to display the plots inline
from matplotlib import pyplot as plt
try:
    ipy = str(type(get_ipython()))
    if 'zmqshell' in ipy or 'terminal' in ipy:
        get_ipython().run_line_magic('matplotlib', 'inline')

except NameError:
    import matplotlib

    #
    # iff not in notebook, active GUI backend for plot.show()
    #
    matplotlib.use('TkAgg')

repo = git.Repo(".", search_parent_directories=True)
rootdir = repo.git.rev_parse("--show-toplevel")


## #### Load project parameters
path = os.path.join(rootdir, "config.json")
with open(path, "r") as file:
    parameters = json.load(file)


## #### Setup work directories
dirs = {'data': os.path.join(rootdir, 'data'),
        'labels': os.path.join(rootdir, 'data', 'labels'),
        'trained model': os.path.join(rootdir, 'models', 'trained')}

for k, v in dirs.items():
    print(f'{k} directory: {v}')
    if not os.path.exists(v):
        os.makedirs(v, exist_ok=True)


## #### Load model checkpoint
path = os.path.join(dirs['trained model'], 'pipeline.config')
pipeline = config_util.get_configs_from_pipeline_file(path)

model = model_builder.build(model_config=pipeline['model'], is_training=False)

ckpt = parameters['trained model']['ckpt']
checkpoint = tf.compat.v2.train.Checkpoint(model=model)
checkpoint.restore(os.path.join(dirs['trained model'], f'ckpt-{ckpt}')).expect_partial()

labelmap = os.path.join(dirs['labels'], 'map.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(labelmap)


## tensorflow object detection method
@tf.function
def detect_fn(image):
    image, shapes = model.preprocess(image)
    prediction = model.predict(image, shapes)
    detections = model.postprocess(prediction, shapes)
    return detections


## object-detection helper
def detect(img):
    tensor = tf.convert_to_tensor(np.expand_dims(img, 0), dtype=tf.float32)
    detections = detect_fn(tensor)

    count = int(detections.pop('num_detections'))
    detections = {key: value[0, :count].numpy() for key, value in detections.items()}

    detections['num_detections'] = count
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    img_objects = img.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(img_objects,
                                                        detections['detection_boxes'],
                                                        detections['detection_classes'] + 1,
                                                        detections['detection_scores'],
                                                        category_index,
                                                        use_normalized_coordinates=True,
                                                        max_boxes_to_draw=5,
                                                        min_score_thresh=.8,
                                                        agnostic_mode=False)

    return img_objects


## #### Image-object detection
ext = parameters['image format']
imgs = glob.glob(os.path.join(dirs['data'], 'test', f'*.{ext}'))
for img in imgs:
    cv_img = cv2.imread(img)
    np_img = np.array(cv_img)

    np_img_objects = detect(np_img)
    plt.imshow(cv2.cvtColor(np_img_objects, cv2.COLOR_BGR2RGB))
    plt.show()
