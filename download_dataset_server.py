
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
    download_dataset("7fae5e0c77d84f69b619a46ab06626ec")