# Pipeline ClearML

# Import necessary libraries
import os
from clearml import Task, Dataset
from clearml.automation import PipelineController
from dotenv import load_dotenv



pipe = PipelineController(
     name='pipeline vits2',
     project='Vits2 Project',
     version='0.0.1',
     add_pipeline_tags=False,
)

pipe.set_default_execution_queue("jobs_urgent")


pipe.add_step(
    name='preprocess_data_mel',
    base_task_project='Vits2 Project',
    base_task_name='Preprocess Vits2 - Meltransform',
)

pipe.add_step(
    name='preprocess_data_filelists',
    parents=["preprocess_data_mel"],
    base_task_project='Vits2 Project',
    base_task_name='Preprocess Vits2 - Filelists',
)

pipe.add_step(
    name='train',
    parents=["preprocess_data_filelists"],
    base_task_project='Vits2 Project',
    base_task_name='Training Vits2',
)


pipe.start()
