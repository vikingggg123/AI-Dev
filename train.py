if __name__ == "__main__":
    from torch.optim.lr_scheduler import StepLR
    from sklearn.metrics import confusion_matrix, classification_report
    import os
    from pathlib import Path
    import data_setup, engine, model, util
    import torchvision
    from torchvision.transforms import v2
    import torch
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, ResNet18_Weights
    from torchvision.transforms import autoaugment, AutoAugmentPolicy


    train_transform = v2.Compose([
    v2.Resize(size=(224, 224)),
    autoaugment.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomRotation(degrees=30),
    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = v2.Compose([
    v2.Resize(size=(224, 224)),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 0.00001 

    dataset_path = Path("LEBINDSV2")
    train_path = dataset_path / "train"
    test_path = dataset_path / "val"

    # Create dataloader for validation set and train set
    train_loader, val_loader, class_name = data_setup.create_dataloader(train_dir=train_path,
                                                                        test_dir=test_path,
                                                                        train_transform=train_transform,
                                                                        test_transform=val_transform,
                                                                        batch_size=BATCH_SIZE)

    print(train_loader)

    # initiate a model

    model = torchvision.models.resnet18(ResNet18_Weights.DEFAULT)

    for param in model.layer4.parameters():
        param.requires_grad = True
    
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3,inplace=True),
        torch.nn.Linear(in_features=model.fc.in_features,out_features=len(class_name),bias=True)
    )

#     class_counts = [train_loader.dataset.targets.count(c) for c in range(6)]  # Count instances per class
#     class_weights = [1.0 / count for count in class_counts]  # Inverse of counts as weights

# # Convert weights to tensor
#     class_weights_tensor = torch.tensor(class_weights)

    
#     loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor,label_smoothing=0.1)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    engine.train(model=model,
                train_dataloader=train_loader, 
                test_dataloader=val_loader, 
                optimizer=optimizer, 
                loss_fn=loss_fn, 
                epochs=EPOCHS,
                schedular=scheduler)

    util.save_model(model=model,
                    directory="model",
                    model_name="Lebin.pth")
