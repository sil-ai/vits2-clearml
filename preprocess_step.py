import os
from clearml import Task, Dataset
from dotenv import load_dotenv

# Get path from dataset_path.txt
with open("dataset_path.txt", "r") as f:
    path = f.read()

# Mel transform preprocessing task
task = Task.create(
    project_name='Vits2 Project',
    task_name='Preprocess Vits2 - Meltransform - Final',
    repo='https://github.com/sil-ai/vits2-clearml.git',
    branch='main',
    script='preprocess/mel_transform.py',
    requirements_file='./requirements.txt',
    docker='alejandroquinterosil/clearml-image:v11',
    argparse_args=[
        ("data_dir", path),
        ("config", path+"datasets/ljs_base/config.yaml")
        ],
    add_task_init_call=True
)

task.execute_remotely(queue_name='jobs_urgent', exit_process=True)
task.close()  # Close the task after execution

# Filelists.py preprocessing task
task = Task.create(
    project_name='Vits2 Project',
    task_name='Preprocess Vits2 - Filelists - Final',
    repo='https://github.com/sil-ai/vits2-clearml.git',
    branch='main',
    script='datasets/ljs_base/prepare/filelists.py',
    requirements_file='./requirements.txt',
    docker='alejandroquinterosil/clearml-image:v11',
    add_task_init_call=True,
    argparse_args=[
        ("data_dir", path),
        ]
)

task.execute_remotely(queue_name='jobs_urgent', exit_process=True)
task.close()  # Close the task after execution