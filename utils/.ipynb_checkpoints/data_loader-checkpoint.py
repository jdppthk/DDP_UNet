import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch import Tensor
import h5py

def worker_init(wrk_id):
    np.random.seed(torch.utils.data.get_worker_info().seed%(2**32 - 1))


def get_data_loader_distributed(params, world_rank):
    fname = params.data_path
    with h5py.File(fname, 'r') as f:
        Fields = f['input_data']['fields'][:,:,:,:].astype(np.float32)
        Residuals = f['target_data']['residual_tensor'][:,:,:,:].astype(np.float32)

    
    Fields = np.moveaxis(Fields, -1, 1)
    Residuals = np.moveaxis(Residuals, -1, 1)
    
    dataset = GetDataset(params, Fields, Residuals)

    train_loader = DataLoader(dataset,
                              batch_size=params.batch_size,
                              num_workers=params.num_data_workers,
                              worker_init_fn=worker_init,
                              pin_memory=torch.cuda.is_available())
    return train_loader

def get_data_loader_distributed_test(params, world_rank):
    fname = params.test_data_path
    with h5py.File(fname, 'r') as f:
        Fields = f['input_data']['fields'][:,:,:,:].astype(np.float32)
        Residuals = f['target_data']['residual_tensor'][:,:,:,:].astype(np.float32)

    
    Fields = np.moveaxis(Fields, -1, 1)
    Residuals = np.moveaxis(Residuals, -1, 1)
    
    dataset = GetDataset(params, Fields, Residuals)

    test_loader = DataLoader(dataset,
                              batch_size=params.batch_size,
                              num_workers=params.num_data_workers,
                              worker_init_fn=worker_init,
                              pin_memory=torch.cuda.is_available())
    return test_loader

class GetDataset(Dataset):
    """Random crops"""
    def __init__(self, params, Fields, Residuals):
        self.Fields = Fields
        self.Residuals = Residuals
        self.Nsamples = Fields.shape[0]

    def __len__(self):
        return self.Nsamples

    def __getitem__(self, idx):
        inp = self.Fields[idx, :, :, :]
        tar = self.Residuals[idx, :, :, :]
        
        tarmean = np.mean(tar, axis = (1,2), keepdims = True)
        tarmin = np.min(tar, axis=(1,2), keepdims = True)
        tarmax = np.max(tar, axis=(1,2), keepdims = True)
        spread = tarmax - tarmin
        
        tar = 2*(tar - tarmean)/spread
        
        return torch.as_tensor(inp), torch.as_tensor(tar)

