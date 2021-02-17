import os

import torch
from torchvision.datasets import VisionDataset
from typing import Any, Callable, Optional, Tuple

# %% Read Metadata

class BarkleyDataset(VisionDataset):
    training_file = 'train.pt'
    test_file = 'test.pt'
    
    def __init__(
            self,
            root: str,
            train: bool = True,
            chaotic: bool = False,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            depth: int = 32,
            time_steps: int = 32
    ) -> None:
        
        t = lambda data:(data.float()+127)/255.
        
        if transform==None:
            transform = t
        if target_transform==None:
            target_transform = t
        super(BarkleyDataset, self).__init__(root, transform=transform, target_transform=target_transform)
        self.train = train  # training set or test set

        
        self.chaotic = chaotic
        self.depth = depth
        self.time_steps = time_steps
        
        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file
        self.X, self.y = torch.load(os.path.join(self.folder, data_file))
        self.X, self.y = self.transform(self.X[:,:time_steps]), self.target_transform(self.y[:,:,:depth])

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (time-series at the surface, dynamic at time=0 till depth=self.depth)
                shape: ([N,T,1,120,120], [N,1,D,120,120]), T and D are choosen in __init__, 
                N=1024 on training-set and N=512 on test-set
        """
        X, y = self.X[index], self.y[index]
        
        #if self.transform is not None:
        #    X = self.transform(X)

        #if self.target_transform is not None:
        #    y = self.target_transform(y)

        return X, y

    def __len__(self) -> int:
        return len(self.X)
    
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body.append("Simulation type: {}".format("chaotic" if self.chaotic else "concentric"))
        body.append("Max. depth: {}".format(self.depth))
        body.append("Number of time-steps: {}".format(self.time_steps))
        body += self.extra_repr().splitlines()
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    @property
    def folder(self) -> str:
        if self.chaotic:
            folder = ""#"chaotic"
        else:
            folder = ""#"concentric"
            
        return os.path.join(self.root, folder)

    def _check_exists(self) -> bool:       
        return (os.path.exists(os.path.join(self.folder, self.training_file)) and
                os.path.exists(os.path.join(self.folder, self.test_file)))

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")
    

    
    
    
    
    
    
    
    