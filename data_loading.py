import h5py
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split

class H5Dataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None
        with h5py.File(file_path, 'r') as h5f:
            self.length = len(h5f['images'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.file_path, 'r')

        image = self.file['images'][idx]
        depth = self.file['depths'][idx]
        
        image = torch.tensor(image, dtype=torch.uint8)
        depth = torch.tensor(depth, dtype=torch.float32)

        return image, depth
    
def create_dataloaders(datasets: list[H5Dataset], batch_size: int = 32):
    test_split = 0.1
    val_split = 0.1
    train_sets = []
    test_sets = []
    val_sets = []
    
    for ds in datasets:
        lenght = len(ds)
        test_amount, val_amount = int(lenght * test_split), int(lenght * val_split)
        train_amount = lenght - (test_amount+val_amount)
        
        train_set, test_set, val_set = random_split(ds, [train_amount, test_amount, val_amount])
        
        train_sets.append(train_set)
        test_sets.append(test_set)
        val_sets.append(val_set)
        
    train_dataset = ConcatDataset(train_sets)
    test_dataset = ConcatDataset(test_sets)
    val_dataset = ConcatDataset(val_sets)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader