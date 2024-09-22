import datasets

DATASET_SOURCE = 'maharshipandya/spotify-tracks-dataset'


def download_dataset(to_folder='data/raw'):
    ds = datasets.load_dataset(DATASET_SOURCE, split='train')
    ds.save_to_disk(to_folder)


def load_dataset(from_folder='data/raw') -> datasets.Dataset:
    return datasets.load_from_disk(from_folder)
