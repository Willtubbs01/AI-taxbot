from torch.utils.data import Dataset

from taxmoe.data.token_block_dataset import TokenBlockDataset


class TinyDataset(Dataset):
    def __init__(self):
        self.rows = [
            {"record_id": "a", "input_ids": list(range(10)), "labels": list(range(10))},
            {"record_id": "b", "input_ids": [20, 21, 22], "labels": [20, 21, 22]},
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def test_token_block_dataset_preserves_all_tokens_without_truncation():
    ds = TokenBlockDataset(TinyDataset(), sequence_length=4)
    assert len(ds) == 4
    assert ds.total_tokens == 13
    assert [len(ds[i]["input_ids"]) for i in range(len(ds))] == [4, 4, 2, 3]
    flattened = [token for i in range(len(ds)) for token in ds[i]["input_ids"]]
    assert flattened == list(range(10)) + [20, 21, 22]

class RemainderOneDataset(Dataset):
    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {"record_id": "r", "input_ids": list(range(9)), "labels": list(range(9))}


def test_token_block_dataset_avoids_one_token_tail():
    ds = TokenBlockDataset(RemainderOneDataset(), sequence_length=4)
    assert [len(ds[i]["input_ids"]) for i in range(len(ds))] == [4, 3, 2]
    assert [token for i in range(len(ds)) for token in ds[i]["input_ids"]] == list(range(9))
