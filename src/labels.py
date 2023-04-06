# Label images

import os
import git
import glob
import xml.etree.ElementTree as ET

repo = git.Repo(".", search_parent_directories=True)
rootdir = repo.git.rev_parse("--show-toplevel")


class Objects:
    def __init__(self, obj):
        bndbox = obj.find("bndbox")
        self.name = obj.findtext("name")
        self.pose = obj.findtext("pose")
        self.truncated = int(obj.findtext("truncated"))
        self.difficult = int(obj.findtext("difficult"))
        self.bndbox = {
            "xmin": int(bndbox.findtext("xmin")),
            "ymin": int(bndbox.findtext("ymin")),
            "xmax": int(bndbox.findtext("xmax")),
            "ymax": int(bndbox.findtext("ymax"))
        }


class Label():
    def __init__(self, xmlfile):
        self.tree = ET.parse(xmlfile)
        self.objects = [Objects(label)
                        for label in self.tree.findall("object")]

    def folder(self):
        return self.tree.findtext("folder")

    def filename(self):
        return self.tree.findtext("filename")

    def path(self):
        return self.tree.findtext("path")

    def database(self):
        return self.tree.findtext("source/database")

    def size(self):
        size = self.tree.find("size")

        if size is None:
            return None

        return {"width": int(size.findtext("width")),
                "height": int(size.findtext("height")),
                "depth": int(size.findtext("depth"))}

    def segmented(self):
        return int(self.tree.findtext("segmented"))

    def objects(self):
        objects = []

        for obj in self.tree.findall("object"):
            bndbox = obj.find("bndbox")

            if bndbox is None:
                return None

            objects.append({"name": obj.findtext("name"),
                            "pose": obj.findtext("pose"),
                            "truncated": int(obj.findtext("truncated")),
                            "difficult": int(obj.findtext("difficult")),

                            "bndbox": {"xmin": int(bndbox.findtext("xmin")),
                           "ymin": int(bndbox.findtext("ymin")),
                                       "xmax": int(bndbox.findtext("xmax")),
                                       "ymax": int(bndbox.findtext("ymax"))}
                            })
        return objects


#
# setup work directories
#
dirs = {'raw': os.path.join(rootdir, 'data', 'raw'),
        'labels': os.path.join(rootdir, 'data', 'labels')}

#
# generate label map
#
labels = []
classes = set()

files = glob.glob(os.path.join(dirs['raw'], "*.xml"))
offset = 1 # id: 0 is reserved for background 
for file in files:
    for annotation in Label(file).objects:
        name = annotation.name

        if name not in classes:
            label = {'id': len(labels) + offset, 'name': name}
            labels.append(label)
            classes.add(name)

labelmap = os.path.join(dirs['labels'], 'map.pbtxt')
with open(labelmap, 'w') as file:
    for label in [label for label in labels]:
        file.write('item { \n')
        file.write('\tname:\'{}\'\n'.format(label['name']))
        file.write('\tid:{}\n'.format(label['id']))
        file.write('}\n')

assert os.path.exists(labelmap), f"Unable to resolve: {labelmap}"
