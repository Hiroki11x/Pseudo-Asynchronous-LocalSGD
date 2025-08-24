from transformers import GPT2Tokenizer
import torch, os
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from datasets import load_dataset, load_from_disk
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class DSet(Dataset):
    def __init__(self, text, config):
        self.tok = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
        self.tok.pad_token = self.tok.eos_token
        self.text = text
        self.config = config

    def __getitem__(self, idx):
        text = self.text[idx]['text']
        src = self.tok(
            text, truncation=True, padding="max_length", max_length=self.config.max_length, return_tensors="pt"
        )
        return {
            "input_ids": src["input_ids"].squeeze(0).long(),
            "attention_mask": src["attention_mask"].squeeze(0).long(),
            "labels": src["input_ids"].squeeze(0).long(),
        }

    def __len__(self):
        return len(self.text)

class DataModule:
    def __init__(self, config):
        self.config = config
        self.warmup_batch = 0

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def collate_fn(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]

        max_length = max([x.size(0) for x in input_ids])
        for i in range(len(input_ids)):
            if input_ids[i].size(0) < max_length:
                input_ids[i] = torch.cat([input_ids[i], torch.zeros(max_length - input_ids[i].size(0), dtype=torch.long)])
                attention_mask[i] = torch.cat([attention_mask[i], torch.zeros(max_length - attention_mask[i].size(0), dtype=torch.long)])
                labels[i] = torch.cat([labels[i], torch.full((max_length - labels[i].size(0),), -100, dtype=torch.long)])

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "labels": labels.long()
        }

    def setup(self):
        if self.config.cluster == "y_cluster":
            dataset_dir = os.environ.get("HINADORI_LOCAL_SCRATCH", "./")
        else:
            dataset_dir = os.environ.get("DATA_DIR_PATH", "./")

        print(f"Cluster name: {self.config.cluster}")
        print(f"Dataset dir: {dataset_dir}")
        
        dset = load_dataset(self.config.dataset_name, split="train", cache_dir=dataset_dir)
        random.seed(self.config.seed)
        dset = dset.shuffle(seed=self.config.seed)

        split_datasets = dset.train_test_split(test_size=self.config.test_size)

        self.trn_dset = DSet(split_datasets['train'], self.config)
        self.trn_dset = ConcatDataset([Subset(self.trn_dset, range(len(self.trn_dset) - self.warmup_batch * 12, len(self.trn_dset))), self.trn_dset])
        self.val_dset = DSet(split_datasets['test'], self.config)

    def get_dataloader(self, dataset, is_train=True):
        if is_train:
            shuffle = True
        else:
            shuffle = False

        data_generator = torch.Generator().manual_seed(self.config.seed + self.config.rank)

        return DataLoader(
            dataset,
            num_workers=self.config.num_dataload_workers,
            batch_size=self.config.batch_size,
            collate_fn=self.collate_fn,
            seed_worker=self.seed_worker,
            generator=data_generator,
            shuffle=shuffle
        )
