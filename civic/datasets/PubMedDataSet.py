from torch.utils.data import Dataset


class CivicEvidenceDataSet(Dataset):
    @staticmethod
    def full_train_dataset(tokenizer, tokenizer_max_length):
        raise NotImplementedError

    @staticmethod
    def full_test_dataset(tokenizer, tokenizer_max_length):
        raise NotImplementedError

    def __init__(self):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError
