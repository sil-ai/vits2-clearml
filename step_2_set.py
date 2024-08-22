from dotenv import load_dotenv
from clearml import StorageManager, Dataset, Task
import os

load_dotenv()

task = Task.init(project_name='Vits2 Project', task_name='Set Dataset')
aws_region = os.getenv('AWS_REGION')
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')

task.set_base_docker(
                docker_image="alejandroquinterosil/clearml-image:v11",
                docker_arguments=[
                f"--env AWS_REGION={aws_region}",
                f"--env AWS_ACCESS_KEY_ID={aws_access_key_id}",
                f"--env AWS_SECRET_ACCESS_KEY={aws_secret_access_key}"
                ]
            )

task.execute_remotely(queue_name='jobs_urgent', exit_process=True)

dataset = Dataset.get(dataset_id="6ec7f9f4265049039400b65a889199a4")

path = dataset.get_mutable_local_copy(
    target_folder="./sil-vits2",
    overwrite=True
)

print("Path Location: ", path)
# Define the path and the link name
link_name = 'DUMMY1'
target_path = "./datasets-vits2/wavs"

# Create the symbolic link
if not os.path.islink(link_name):
    os.symlink(target_path, link_name)