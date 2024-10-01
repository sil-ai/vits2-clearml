# Pipeline ClearML

# Import necessary libraries
import os
from clearml import Task, Dataset
from clearml.automation import PipelineController
from dotenv import load_dotenv
import argparse

load_dotenv()

gh_user = os.getenv('GITHUB_USERNAME')
gh_token = os.getenv('GITHUB_TOKEN')

# Argsparser
# Add arguments dataset_path, bucket_name, and dataset_name
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_id")
args = parser.parse_args()


pipe = PipelineController(
     name='pipeline vits2',
     project='Vits2 Project',
     version='0.0.2',
     add_pipeline_tags=False,
)


pipe.set_default_execution_queue("jobs_urgent")

pipe.add_parameter(
    name='dataset_id',
    default=args.dataset_id,
)

pipe.add_step(
    name='preprocess_data_mel',
    base_task_project='Vits2 Project',
    base_task_name='Preprocess Vits2 - Meltransform',
    parameter_override={'General/dataset_id': '${pipeline.dataset_id}'},
)

pipe.add_step(
    name='preprocess_data_filelists',
    parents=["preprocess_data_mel"],
    base_task_project='Vits2 Project',
    base_task_name='Preprocess Vits2 - Filelists',
    parameter_override={'General/dataset_id': '${preprocess_data_mel.artifacts.new_dataset_id}'},
)

pipe.add_step(
    name='train',
    parents=["preprocess_data_filelists"],
    base_task_project='Vits2 Project',
    base_task_name='Training Vits2',
    parameter_override={'General/dataset_id': '${preprocess_data_filelists.artifacts.new_dataset_id}'},
)


pipe.start()
