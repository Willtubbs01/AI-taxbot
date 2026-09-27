from __future__ import annotations

from torch.utils.data import Dataset


class TokenBlockDataset(Dataset):
    """Deterministically split cached training records into bounded token blocks.

    Long records are sliced consecutively and every token is preserved. A final
    one-token tail is avoided by borrowing one token from the previous block,
    because causal-LM training needs at least two tokens to produce a shifted
    next-token target.
    """

    def __init__(self, base: Dataset, sequence_length: int):
        if sequence_length < 3:
            raise ValueError("TRAIN-SEQUENCE-LENGTH-INVALID")
        self.base = base
        self.sequence_length = int(sequence_length)
        self.blocks: list[tuple[int, int, int]] = []
        self.total_tokens = 0
        for record_index in range(len(base)):
            example = base[record_index]
            n = len(example["input_ids"])
            if n != len(example["labels"]):
                raise ValueError("CACHE-LABEL-LENGTH-MISMATCH")
            if n < 2:
                raise ValueError(f"TRAIN-RECORD-TOO-SHORT:{example.get('record_id', record_index)}")
            self.total_tokens += n
            start = 0
            while n - start > self.sequence_length:
                end = start + self.sequence_length
                # Do not leave a single token as the final block.
                if n - end == 1:
                    end -= 1
                self.blocks.append((record_index, start, end))
                start = end
            self.blocks.append((record_index, start, n))

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int):
        record_index, start, end = self.blocks[index]
        source = self.base[record_index]
        labels = source["labels"][start:end]
        out = {
            "record_id": f"{source['record_id']}#tok={start}:{end}",
            "input_ids": source["input_ids"][start:end],
            "labels": labels,
            "token_count": end - start,
            "supervised_token_count": sum(1 for x in labels if x != -100),
        }
        if "ngram_feature_ids" in source:
            out["ngram_feature_ids"] = source["ngram_feature_ids"][start:end]
        return out
