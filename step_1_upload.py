# create example dataset
import argparse
import os
from clearml import StorageManager, Dataset, Task

task = Task.init(project_name='Vits2 Project', task_name='Upload Dataset')

task.execute_remotely(queue_name='jobs_urgent', exit_process=True)

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

