import h5py
import torch
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split, Sampler

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
    
def create_random_dataloaders(datasets: list[H5Dataset], batch_size: int = 32):
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
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader, val_loader

class EqualSampler(Sampler):
    def __init__(self, dataset_sizes: list[int], samples_per_dataset: int):
        self.dataset_sizes = dataset_sizes
        self.samples_per_dataset = samples_per_dataset
        self.max_size = max(dataset_sizes)
        # Calculate starting index for each dataset in concatenated dataset
        self.dataset_starts = [0]
        for size in dataset_sizes[:-1]:
            self.dataset_starts.append(self.dataset_starts[-1] + size)
            
    def __iter__(self):
        dataset_indices = []
        for start, size in zip(self.dataset_starts, self.dataset_sizes):
            indices = list(range(start, start + size))
            random.shuffle(indices)
            
            # If this dataset is smaller than the largest one, repeat indices
            if size < self.max_size:
                # Calculate how many full repeats we need
                repeats = self.max_size // size
                remainder = self.max_size % size
                
                # Repeat the full list as needed
                repeated_indices = indices * repeats
                
                # Add the remainder indices (shuffled again to avoid pattern at the end)
                remainder_indices = indices.copy()
                random.shuffle(remainder_indices)
                repeated_indices.extend(remainder_indices[:remainder])
                
                dataset_indices.append(repeated_indices)
            else:
                dataset_indices.append(indices)
        
        for batch_idx in range(self.__len__()):
            batch = []
            start_idx = batch_idx * self.samples_per_dataset
            end_idx = start_idx + self.samples_per_dataset
            
            # Take the next chunk from each dataset
            for dataset_idx in range(len(self.dataset_sizes)):
                batch.extend(dataset_indices[dataset_idx][start_idx:end_idx])
            
            # Shuffle within the batch
            random.shuffle(batch)
            yield batch
            
    def __len__(self):
        return self.max_size // self.samples_per_dataset

def create_equal_dataloaders(datasets: list[H5Dataset], batch_size: int = 30):
    assert batch_size % len(datasets) == 0, "batch_size must be a multiple of the numer of datasets"
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
    
    train_loader = DataLoader(train_dataset, batch_sampler=EqualSampler([len(d) for d in train_sets], batch_size//len(datasets)))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, test_loader, val_loader