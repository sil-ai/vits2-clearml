
from clearml import Dataset

def download_dataset(id):
    dataset = Dataset.get(
        dataset_id=id
    )

    path = dataset.get_mutable_local_copy(
        target_folder="./datasets-vits2",
        overwrite=True
    )
    print("Path Location: ", path)

if __name__ == "__main__":
    download_dataset("d087dddf638d4ba3a616cebe3fd02454")