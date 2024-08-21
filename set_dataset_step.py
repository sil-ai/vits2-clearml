import os
from clearml import StorageManager, Dataset, Task

task = Task.init(project_name='Vits2 Project', task_name='Set Dataset')


dataset = Dataset.get(
  dataset_id='62f5075579034a659d573e2ad59e624e'
)

path = dataset.get_mutable_local_copy(
    target_folder="./sil-vits2",
    overwrite=True
)

print("Path Location: ", path)


# Upload artifact
task.upload_artifact('dataset_path', artifact_object=path)
task.upload_artifact('config', artifact_object=path+"datasets/ljs_base/config.yaml")

# Define the path and the link name
link_name = 'DUMMY1'
target_path = "./datasets-vits2/wavs"

# Create the symbolic link
if not os.path.islink(link_name):
    os.symlink(target_path, link_name)