# Partition images

import os
import git
import json
import glob
import shutil
from math import ceil

repo = git.Repo(".", search_parent_directories=True)
rootdir = repo.git.rev_parse("--show-toplevel")

path = os.path.join(rootdir, 'config.json')
with open(path, "r") as file:
    parameters = json.load(file)

def process(rawdir, traindir, testdir, extensions=(".jpg", ".xml")):

    def cleanup(dir):
        for filename in os.listdir(dir):
            path = os.path.join(dir, filename)
            ext = os.path.splitext(filename)[1].lower()
            if os.path.isfile(path) and ext in extensions:
                os.remove(path)

    cleanup(traindir)
    cleanup(testdir)

    files = {}
    for ext in extensions:
        files[ext] = glob.glob(os.path.join(rawdir, f"*{ext}"))

    data = {}
    for ext, filenames in files.items():
        names = [os.path.splitext(os.path.basename(name))[0]
                 for name in filenames]
        data[ext] = set(names)

    #
    # find matching image-label names
    #
    matching = set.intersection(*data.values())

    #
    # check sets in data.values() are equal to the matching set
    #
    errmsg = "Some images do not have corresponding labels."
    assert all(matching == filenames for filenames in data.values()), errmsg

    index = ceil(len(matching) * parameters['split'])
    for enum, filename in enumerate(matching):
        raw = {ext: os.path.join(rawdir, f"{filename}{ext}") for ext in extensions}
        dst = traindir if enum < index else testdir

        for ext in extensions:
            shutil.copy(raw[ext], os.path.join(dst, f"{filename}{ext}"))


#
# setup work directories
#
dirs = {'raw': os.path.join(rootdir, 'data', 'raw'),
        'test': os.path.join(rootdir, 'data', 'test'),
        'train': os.path.join(rootdir, 'data', 'train')}

for k, v in dirs.items():
    if not os.path.exists(v):
        os.makedirs(v, exist_ok=True)

#
# partition training and test data
#
ext = parameters['image format']
process(dirs['raw'], dirs['train'], dirs['test'], extensions=(f".{ext}", ".xml"))
