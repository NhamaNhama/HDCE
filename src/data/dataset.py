import torch
from torch.utils.data import Dataset

class HDCETextGraphDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: List[dict]
          {
            "text": "some input text",
            "label": int,
            "graph_data": { ... },
            "external_concept": "apple"
          }
        """
        # data_listをそのまま保持
        self.data_list = data_list
    
    def __len__(self):
        # データ数を返す
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # インデックスで指定されたデータを返す
        return self.data_list[idx] 