# create example dataset
from clearml import StorageManager, Dataset

# Create a dataset with ClearML`s Dataset class
dataset = Dataset.create(
    dataset_project="DemoData",
    dataset_name="LJSpeech-1.1",
    output_uri="s3://tts-training-examples/"
)

# add the example csv
dataset.add_files(path='./LJSpeech-1.1')


# Upload dataset to ClearML server (customizable)
dataset.upload()

# commit dataset changes
dataset.finalize()