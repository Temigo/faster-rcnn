import os
import os.path as osp
import sys

if not 'RCNNDIR' in os.environ:
    print('RCNNDIR not in shell env. (did you source setup.sh?)')
    raise ImportError

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
#lib_path = osp.join(this_dir, '..', 'lib')
lib_path = osp.join(os.environ['RCNNDIR'],'lib')
add_path(lib_path)

#coco_path = osp.join(this_dir, '..', 'data', 'coco', 'PythonAPI')
coco_path = osp.join(os.environ['RCNNDIR'], 'data', 'coco', 'PythonAPI')
add_path(coco_path)

