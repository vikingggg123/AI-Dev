from typing import Tuple, List, Dict
import torch
import torch.utils.data.dataloader
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, classification_report

def early_stopping(train_loss, validation_loss, min_delta, tolerance):
    counter = 0
    if (validation_loss - train_loss) > min_delta:
        counter +=1
        if counter >= tolerance:
          return True
        
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device = "cpu") -> Tuple[float,float]:
    model.train()

    train_loss, train_acc = 0, 0

    for input, targets in dataloader:
        input, targets = input.to(device), targets.to(device)

        y_pred = model(input)

        loss = loss_fn(y_pred,targets)
        train_loss += loss

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == targets).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    print(f"Accuracy: {train_acc}")
    print(f"loss: {train_loss}")
    
    return train_loss, train_acc
        

    pass
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device = "cpu"):
    preds, labels = [],[]
    with torch.inference_mode():

        val_loss, val_acc = 0,0

        for input, target in dataloader:

            input, target = input.to(device), target.to(device)

            y_pred = model(input)
            preds.extend(y_pred.argmax(1).numpy())
            labels.extend(target.numpy())

            loss = loss_fn(y_pred,target)
            val_loss += loss

            test_pred_labels = y_pred.argmax(dim=1)
            val_acc += ((test_pred_labels == target).sum().item()/len(test_pred_labels))

        val_loss /= len(dataloader)
        val_acc /= len(dataloader)
        print(f"val_acc:{val_acc}")
        print(f"val_loss:{val_loss}")
        print(confusion_matrix(y_true=labels, y_pred=preds))
        print(classification_report(labels,preds))
        return val_loss, val_acc


    

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          schedular: torch.optim.Optimizer,
          device: torch.device = "cpu",
          ) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    Passes a target PyTorch models through train_step() and test_step()
    functions for a number of epochs, training and testing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              test_loss: [...],
              test_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              test_loss: [1.2641, 1.5706],
              test_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []
    }

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        print(f"epoch: {epoch}\n ----------")
        train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        schedular.step()
        
        if early_stopping(train_loss=train_loss, validation_loss=test_loss, min_delta=10, tolerance=20):
            break

        # Print out what's happening
        print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # Return the filled results at the end of the epochs
    return model