import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd

def create_data_loader(df: pd.DataFrame, batch_size: int = 64):
    """
    Converte un DataFrame Pandas in un Tensor PyTorch e restituisce il DataLoader.
    """
    data_tensor = torch.tensor(df.values, dtype=torch.float32)
    dataset = TensorDataset(data_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Restituisce sia il loader che la dimensione delle feature (input_dim)
    return train_loader, data_tensor.shape[1]