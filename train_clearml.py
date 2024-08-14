from clearml import Task, Dataset



task = Task.create(
    project_name='Vits2 Project',
    task_name='Training Vits2',
    repo='https://github.com/daniilrobnikov/vits2.git',
    branch='main',
    script='train.py',
    requirements_file='./requirements.txt',
    docker='alejandroquinterosil/clearml-image:v11',
    argparse_args=[
        ("config", "datasets/ljs_base/config.yaml"),
        ("model", "ljs_base")
        ],
    add_task_init_call=True
)

task.set_script(
    repository='https://github.com/daniilrobnikov/vits2.git',
    branch='main',
    entry_point='train.py'
)

task.execute_remotely(queue_name='jobs_urgent', exit_process=True)

