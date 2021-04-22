from torch.utils.data import Dataset, ConcatDataset

from .mot_reid import MOTreID


class MOTreIDWrapper(Dataset):
    """A Wrapper class for MOTSiamese.

    Wrapper class for combining different sequences into one dataset for the MOTreID
    Dataset.
    """

    def __init__(self, split, kwargs):
        train_mot15_sequences = ['ADL-Rundle-6', 'ADL-Rundle-8', 'ETH-Bahnhof',
                                 'ETH-Pedcross2', 'ETH-Sunnyday', 'KITTI-13',
                                 'KITTI-17', 'PETS09-S2L1', 'TUD-Campus',
                                 'TUD-Stadtmitte', 'Venice-2']
        train_mot16_sequences = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09',
                                 'MOT16-10', 'MOT16-11', 'MOT16-13']
        train_mot17_sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09',
                                 'MOT17-10', 'MOT17-11', 'MOT17-13']
        train_mot20_sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
        train_debug_sequences = ['MOT17-02']

        if split == "train_mot15":
            sequences = train_mot15_sequences
        elif split == "train_mot16":
            sequences = train_mot16_sequences
        elif split == "train_mot17":
            sequences = train_mot17_sequences
        elif split == "train_mot20":
            sequences = train_mot20_sequences
        elif split == "train_debug":
            sequences = train_debug_sequences
        else:
            raise NotImplementedError("MOT split not available.")

        dataset = []
        for seq in sequences:
            # dataset.append(MOTreID(seq, split=split, **kwargs))
            dataset.append(MOTreID(seq, **kwargs))

        self.split = ConcatDataset(dataset)

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx):
        return self.split[idx]
