import numpy as np
import os
from os.path import join, basename, dirname
import torch
import glob
import soundfile as sf
from utils.text import text_to_sequence

# paired text and content units
class Text2UnitDataset(torch.utils.data.Dataset):
    def __init__(self, txt_paths, feature_type, split="train", max_in_len=200, min_in_len=15, max_out_len=600, ratio=1):
        super().__init__()
        assert feature_type in ["hubert"]
        self.split = split
        self.feature_type = feature_type

        data_dict = []
        in_lens = []
        out_lens = []

        prev_len = 0

        for j, txt_path in enumerate(txt_paths):
            with open(txt_path, "r") as f:
                for i, line in enumerate(f):
                    utterance_dict = eval(line.strip("\n"))
                    text_len = len(text_to_sequence(utterance_dict["transcription"]))
                    unit_len = len(utterance_dict[feature_type].split(" "))
                    
                    if text_len < min_in_len or text_len > max_in_len:
                        continue
                        
                    if unit_len > (max_out_len - 2):
                        continue

                    in_lens.append(text_len)
                    out_lens.append(unit_len)
                    data_dict.append(utterance_dict)

            cur_len = len(data_dict) - prev_len
            if split == "val":
                d_len = cur_len // 2
                in_lens = in_lens[:prev_len + d_len]
                out_lens = out_lens[:prev_len + d_len]
                data_dict = data_dict[:prev_len + d_len]
            prev_len = len(data_dict)

            if j == 0 and len(txt_paths) == 2:
                ljspeech_len = len(data_dict)

        if len(txt_paths) == 1:
            ljspeech_len = 0

        in_lens = np.array(in_lens)
        out_lens = np.array(out_lens)
        data_dict = np.array(data_dict)

        if split == "train" and ratio < 1:
            dataset_len = ljspeech_len + int((len(data_dict) - ljspeech_len) * ratio)
            in_lens = in_lens[:dataset_len]
            out_lens = out_lens[:dataset_len]
            data_dict = data_dict[:dataset_len]
        
        if split != "train":
            order = np.argsort(-out_lens)
            in_lens = in_lens[order]
            out_lens = out_lens[order]
            data_dict = data_dict[order]

        self.in_lens = in_lens
        self.out_lens = out_lens
        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict)
        
    def __getitem__(self, idx):
        text = np.array(text_to_sequence(self.data_dict[idx]["transcription"]))
        # 0 for <pad>, 1 for <start>, 2 for <end>
        unit = np.array([int(u) for u in self.data_dict[idx][self.feature_type].split(" ")]) + 3
        unit = np.pad(unit, (1,1))
        unit[0] += 1
        unit[-1] += 2

        return {"text": text, "unit": unit}

# pad data in each batch
class Collator:
    def __init__(self):
        self.mode = "seq2seq" # some random thing, not used

    def collate(self, minibatch):
        max_in_len = np.max([len(data["text"]) for data in minibatch])
        max_out_len = np.max([len(data["unit"]) for data in minibatch])

        for data in minibatch:
            data["text"] = np.pad(data["text"], (0, max_in_len - len(data["text"])))
            data["unit"] = np.pad(data["unit"], (0, max_out_len - len(data["unit"])))

        text = torch.from_numpy(np.stack([data['text'] for data in minibatch]))
        unit = torch.from_numpy(np.stack([data['unit'] for data in minibatch]))

        return {
            'text': text,
            'unit': unit,
        }
    
class Collator_v2:
    def __init__(self, max_in_len=200):
        self.mode = "seq2seq" # some random thing, not used
        self.max_in_len = max_in_len

    def collate(self, minibatch):
        max_out_len = np.max([len(data["unit"]) for data in minibatch])

        for data in minibatch:
            data["text"] = np.pad(data["text"], (0, self.max_in_len - len(data["text"])))
            data["unit"] = np.pad(data["unit"], (0, max_out_len - len(data["unit"])))

        text = torch.from_numpy(np.stack([data['text'] for data in minibatch]))
        unit = torch.from_numpy(np.stack([data['unit'] for data in minibatch]))

        return {
            'text': text,
            'unit': unit,
        }

def from_path(txt_path, batch_size, split="train", num_workers=16, is_distributed=False):
    dataset = Text2UnitDataset(txt_path, "hubert")
    
    max_in_len = np.max(dataset.in_lens)
    max_out_len = np.max(dataset.out_lens)
    
    shuffle = not is_distributed if split == "train" else False
    if split != "train":
        is_distributed = False
        
    print(f"Shuffle is {shuffle}")
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=Collator().collate,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        drop_last=True)

def from_path_v2(txt_path, batch_size, split="train", max_in_len=200, min_in_len=15, max_out_len=512, num_workers=16, ratio=1, is_distributed=False):
    dataset = Text2UnitDataset(txt_path, "hubert", split=split, max_in_len=max_in_len, min_in_len=min_in_len, max_out_len=max_out_len, ratio=ratio)
    
    # max_in_len = np.max(dataset.in_lens)
    # max_out_len = np.max(dataset.out_lens)
    
    shuffle = not is_distributed if split == "train" else False
    if split != "train":
        is_distributed = False
        
    print(f"Shuffle is {shuffle}")
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=Collator_v2(max_in_len=max_in_len).collate,
        shuffle=shuffle,
        num_workers=num_workers,
        sampler=DistributedSampler(dataset) if is_distributed else None,
        drop_last=True)

if __name__ == '__main__':

    dataset = Text2UnitDataset("/u/junkaiwu/ECE537_Project/datasets/LJSpeech/hubert100/train_t.txt", "hubert")

    for i, d in enumerate(dataset):
        print(d)

        if i == 1:
            break

    print(np.min(dataset.in_lens), np.max(dataset.in_lens), np.mean(dataset.in_lens), np.median(dataset.in_lens), np.std(dataset.in_lens))
    print(np.min(dataset.out_lens), np.max(dataset.out_lens), np.mean(dataset.out_lens), np.median(dataset.out_lens), np.std(dataset.out_lens))

    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        collate_fn=Collator().collate,
        shuffle=True,
        num_workers=4)

    for data in dl:
        print(data["text"])
        print(data["unit"])

        print(data["text"].shape)
        print(data["unit"].shape) 

        break


