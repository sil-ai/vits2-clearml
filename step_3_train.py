import os
from clearml import Task, Dataset
from dotenv import load_dotenv

load_dotenv()


task = Task.create(
    project_name='Vits2 Project',
    task_name='Training Vits2',
    repo='https://github.com/sil-ai/vits2-clearml.git',
    branch='main',
    script='train.py',
    requirements_file='./requirements.txt',
    docker='alejandroquinterosil/clearml-image:v11',
    argparse_args=[
        ("model", "ljs_base")
        ],
    add_task_init_call=True
)

aws_region = os.getenv('AWS_REGION')
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

task.set_base_docker(
                docker_image="alejandroquinterosil/clearml-image:v11",
                docker_arguments=[
                f"--env AWS_REGION={aws_region}",
                f"--env AWS_ACCESS_KEY_ID={aws_access_key_id}",
                f"--env AWS_SECRET_ACCESS_KEY={aws_secret_access_key}",
                "--shm-size 8g"]
            )

task.execute_remotely(queue_name='jobs_urgent', exit_process=True)

