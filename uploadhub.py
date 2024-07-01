# Model Conversion
from ultralytics import YOLO

model = YOLO('checkpoint/train3/weights/best.pt')

model.export(format='onnx')

# Upload
from huggingface_hub import HfApi, HfFolder, Repository

username = ''
repo_name = ''

repo_url = f'https://huggingface.co/{username}/{repo_name}'
local_dir = './' + repo_name
repo = Repository(local_dir, clone_from=repo_url)

# Move the ONNX model and other necessary files to the local repository directory
import shutil

shutil.copy('checkpoint/train3/weights/best.onnx', local_dir)
# Add other files if necessary
# shutil.copy('README.md', local_dir)
# shutil.copy('config.yaml', local_dir)

repo.git_add(auto_lfs_track=True)
repo.git_commit('Add YOLOv8 ONNX model')
repo.git_push()