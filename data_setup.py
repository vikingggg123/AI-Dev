import os
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

NUM_WORKER = os.cpu_count()

def create_dataloader(train_dir : str,
                      test_dir : str,
                      train_transform: v2.Compose,
                      test_transform: v2.Compose,
                      batch_size: int,
                      num_worker: int=NUM_WORKER,
                      ):
    
    """this function prepare a dataset to be ready to use, with args of:
            train directory: The path of the training directory
            test directory: the path of the test directory or the validation directory ( both can be use )
            transform: what transform apply to the input images.
            batch size: the amount of image going to be fed through the training loop for 1 cycle
            transfer model
            
        and it will return:
        A tuple of (train dataloader, test dataloader, class name)
            where class name is the list of target classes
            example of using this function
                
                train_loader, test_loader, class_name = create_dataloader(train_dir = "Path/train,
                                                                          test_dir = "Path/test,
                                                                          transform = auto_transform,
                                                                          batch_size = 32,
                                                                          num_worker = NUM_WORKER)"""
    
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=train_transform)
    test_data = datasets.ImageFolder(root=test_dir,
                                     transform=test_transform)
    
    class_name = train_data.classes

    # Now after we got the big datasets, we have to put it into batches or what we call a dataloader

    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_worker,
                              pin_memory=True)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_worker,
                             pin_memory=True)
    
    return train_loader, test_loader, class_name
    
