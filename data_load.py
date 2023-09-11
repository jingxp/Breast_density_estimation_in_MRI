import numpy as np
from torch.utils import data as torch_data

def load_croped_volumn(path):
    img = np.load(path)
    img_data = img['arr_0']
    return img_data
 
class Dataset(torch_data.Dataset):
    def __init__(self, df, size, data_root, transform, binary = False, dense = False, targets=False):
        self.df = df
        self.targets = targets
        self.dense = dense
        self.binary = binary
        self.transform = transform
        self.data_root = data_root
        if self.dense:
            self.df = self.df[self.df.Label>1]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        path = self.df.loc[index].Filepath
        scan_id = self.df.loc[index].patient_ID +'_'+ self.df.loc[index].EXAM_DATE
        data_path = os.path.join(self.data_root,path)
        data = load_croped_volumn(data_path)
        data = torch.tensor(data)
        data = torch.unsqueeze(data,0)
        data = self.transform(data)
        if self.targets:
            y = torch.tensor(self.df.loc[index].label, dtype=torch.long)
            #########
            # Binary classification
            #########
            
            if self.dense:
                if y == 2:
                    label = torch.tensor(0)
                elif y == 3 :
                    label = torch.tensor(1)        
                    
            elif self.binary:
                if y in [0,1]:
                    label = torch.tensor(0)
                elif y in [2,3]:
                    label = torch.tensor(1)
               
            else:
                label = y
                    
            return {"X": data.float(), "y": label, "id": scan_id}
        else:
            return {"X": data.float(), "id": scan_id}
