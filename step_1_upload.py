# create example dataset
import argparse
import os
from clearml import StorageManager, Dataset, Task

# Add arguments dataset_path, bucket_name, and dataset_name
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="./LJSpeech-1.1")
parser.add_argument("--bucket_name", type=str, default="sil-vits2")
parser.add_argument("--dataset_name", type=str, default="LJSpeech-1.1")
args = parser.parse_args()

# Step 4: Use the parsed arguments in your script
# Create a dataset with ClearML's Dataset class

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




