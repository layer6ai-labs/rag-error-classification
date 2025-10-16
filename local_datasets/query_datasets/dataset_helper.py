import pathlib
from local_datasets.query_datasets.query_dataset import DragonballQueryDataset, QueryDataset, ClapNQQueryDataset

DATASET_CLASSES = {
    "dragonball": DragonballQueryDataset,
    "dragonball-chinese": DragonballQueryDataset,
    "clapnq": ClapNQQueryDataset,
}


def get_query_dataset(
    dataset_name: str,
    queries_file: pathlib.Path,
) -> QueryDataset:
    if dataset_name not in DATASET_CLASSES:
        raise ValueError(f"Dataset {dataset_name} not found in {DATASET_CLASSES.keys()}")
    return DATASET_CLASSES[dataset_name](queries_file)
