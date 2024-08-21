# create example dataset
import argparse
import os
from clearml import StorageManager, Dataset, Task

task = Task.init(project_name='Vits2 Project', task_name='Upload Dataset - Final')


# Step 4: Use the parsed arguments in your script
# Create a dataset with ClearML's Dataset class

dataset = Dataset.create(
    dataset_project="Vits2 - Dev",
    dataset_name="LJSpeech-1.1",
    output_uri=f"s3://sil-vits2/"
)

# Add the example csv
dataset.add_files(path="./LJSpeech-1.1")

# Upload dataset to ClearML server (customizable)
dataset.upload()

# Commit dataset changes
dataset.finalize()

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

task.execute_remotely(queue_name='jobs_urgent', exit_process=True)
