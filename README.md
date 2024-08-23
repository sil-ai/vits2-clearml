# VITS2: SIL-AI ClearML Archirecture

This repo is an adaptation of vits2 model to be able to run on ClearML platform on remote GPUs.

The scripts added to run the model are, each of them are going to be explained on detail
below, to know how to modify them in order to get a desired result:

# ClearML Orchestration Scripts

1. `step_1_upload.py`

How to run it:
```
python step_1_upload.py --dataset_path=./LJSpeech-1.1 --bucket_name=sil-vits2 --dataset_name=LJSpeech-1.1
```

This script is very simple, just takes some dataset you have on local, and upload it to S3,
as well as to ClearML Data Manager. Change the arguments to specify dataset path, buacket where
you want to upload the dataset and the name you want to assign to the dataset. The script
is run locally.

We created bucket sil-vits2 for this repo's task, so you can use that one to store your datasets.

Here is the code performed:
```
dataset = Dataset.create(
    dataset_project="Vits2 - Dev",
    dataset_name= f"{args.dataset_name}",
    output_uri=f"s3://{args.bucket_name}/"
)

# Add the example csv
dataset.add_files(path=f"{args.dataset_path}")

# Upload dataset to ClearML server (customizable)
dataset.upload()

# Commit dataset changes
dataset.finalize()
```


2. `step_2_preprocess.py`

How to run it:
```
python step_2_preprocess.py
```

This script performs the two preprocessing tasks, specified on vits2 repo.

- mel_transform.py
- filelists.py

So naturally this script just invokes them, waits for meltransform to finish, and starts
executing filelists.py.


```
# Mel transform preprocessing task
task = Task.create(
    project_name='Vits2 Project',
    task_name='Preprocess Vits2 - Meltransform',
    repo='https://github.com/sil-ai/vits2-clearml.git',
    branch='main',
    script='preprocess/mel_transform.py',
    requirements_file='./requirements.txt',
    docker='alejandroquinterosil/clearml-image:v11',
    add_task_init_call=True
)

task.execute_remotely(queue_name='jobs_urgent', exit_process=True)
task.close()  # Close the task after execution

# Filelists.py preprocessing task
task = Task.create(
    project_name='Vits2 Project',
    task_name='Preprocess Vits2 - Filelists',
    repo='https://github.com/sil-ai/vits2-clearml.git',
    branch='main',
    script='datasets/ljs_base/prepare/filelists.py',
    requirements_file='./requirements.txt',
    docker='alejandroquinterosil/clearml-image:v12',
    add_task_init_call=True,
)

task.execute_remotely(queue_name='jobs_urgent', exit_process=True)
task.close()  # Close the task after execution
```

NOTE: In the original vits2 repo, you pass the config path and dataset path to meltransform.
But that did not work in clearml, sisnce the scripts can be run on different machine each time,
you have to download it inside the meltransform script, same in filelists.py, and then
take that path. This is currently one of the problems we are working on.

So for example, `parse_args()` on mel_transform.py, that used to parse the args for the paths
of the data and config, got replaced by:

```
def parse_args():
    # Config
    curr_dir = os.getcwd().split('/')
    print("Current Directory: ", curr_dir)
    vits_path = '/'.join(curr_dir)

    # Getting the dataset - data_dir
    dataset = Dataset.get(dataset_id="6ec7f9f4265049039400b65a889199a4")
    path = dataset.get_mutable_local_copy(
        target_folder="./sil-vits2",
        overwrite=True
    )
    link_name = 'DUMMY1'
    target_path = path + "/wavs"
    # Create the symbolic link
    if not os.path.islink(link_name):
        os.symlink(target_path, link_name)

    hparams = get_hparams_from_file(vits_path+"/datasets/ljs_base/config.yaml")
    hparams.data_dir = target_path
    return hparams
```

3. `step_3_train.py`

How to run it:
```
python step_3_train.py
```

Finally this step, trains the model. Some docker arguments are passed, like the credentials,
this is because sometimes clearml will take credentials configured on the server, and
those may not have the necessary access for the S3 bucket or something else, so you
can always use yours.

```
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
```

NOTE: Here happends the same with the arguments, the original repo requires the arguments
config and model, but I only pass model. train.py it self will know where the config path is.
See ´get_hparams()´ in utils/hparams.py.

credits: https://github.com/daniilrobnikov/vits2