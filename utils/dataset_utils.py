# utils/dataset_utils.py

import jieba
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.data = [self.encode(text, vocab, max_len) for text in texts]
        self.labels = labels

    def encode(self, text, vocab, max_len):
        tokens = list(jieba.cut(text))
        ids = [vocab.get(tok, vocab.get("<unk>", 0)) for tok in tokens]
        if len(ids) < max_len:
            ids += [vocab.get("<pad>", 0)] * (max_len - len(ids))
        return ids[:max_len]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
