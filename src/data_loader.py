from datasets import load_dataset

def get_dataset(data_path: str, data_name: str = None, val_size: float = 0.1):

    dataset = load_dataset(data_path, data_name) if data_name else load_dataset(data_path)

   
    train_data = dataset["train"] if "train" in dataset else None
    test_data = dataset["test"] if "test" in dataset else None


    if "validation" in dataset:
        val_data = dataset["validation"]
    else:
        if train_data is not None:
            split = train_data.train_test_split(test_size=val_size)
            train_data = split["train"]
            val_data = split["test"]
        else:
            val_data = None

    return dataset
