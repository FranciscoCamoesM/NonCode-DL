from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        return seq.T, label






def train(model, optimizer, criterion, train_loader, val_loader, device, num_epochs=5000, patience=100, clip_grad=True, N_LABELS=1, lr_scheduler=None):
    epoch_loss = 0.0
    val_epoch_loss = 0.0
    best_val_loss = float('inf')
    best_model_dict = None

    history = []

    with tqdm(range(num_epochs), desc="Training Epochs") as pbar:
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}, Best Val Loss: {best_val_loss:.4f}, Current LR: {optimizer.param_groups[0]['lr']:.3e}")
            pbar.set_postfix(loss=0.0)
            model.train()
            running_loss = 0.0
            for seqs, y in train_loader:
                seqs = seqs.float().to(device)
                labels, classes = y
                labels = labels.float().to(device)


                optimizer.zero_grad()

                class_mask = np.zeros((len(labels), N_LABELS))
                for i, c in enumerate(classes):
                    class_mask[i,c] = 1
                
                class_mask = torch.tensor(class_mask, dtype=torch.float32).to(device)

                outputs = model(seqs) 
                
                labels = labels.unsqueeze(dim=1) * class_mask + (1 - class_mask) * outputs.detach()
                
                loss = criterion(outputs, labels)
                loss.backward()
                if clip_grad:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)  # Gradient clipping
                optimizer.step()

                running_loss += loss.item() * seqs.size(0)

            epoch_loss = running_loss / len(train_loader.dataset)

            with torch.no_grad():
                model.eval()
                val_running_loss = 0.0
                for seqs, y in val_loader:
                    seqs = seqs.float().to(device)
                    labels, classes = y
                    labels = torch.tensor(labels, dtype=torch.float32).to(device)

                    class_mask = np.zeros((len(labels), N_LABELS))
                    for i, c in enumerate(classes):
                        class_mask[i,c] = 1

                    class_mask = torch.tensor(class_mask, dtype=torch.float32).to(device)

                    outputs = model(seqs) * class_mask
                    outputs = outputs.sum(dim=1)  # Sum over the classes

                    loss = criterion(outputs, labels)
                    val_running_loss += loss.item() * seqs.size(0)


                val_epoch_loss = val_running_loss / len(val_loader.dataset)
            if lr_scheduler is not None:
                lr_scheduler.step(val_running_loss)
            

            history.append((epoch_loss, val_epoch_loss))

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                best_model_dict = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break


    #load the best model
    if best_model_dict is not None:
        model.load_state_dict(best_model_dict)
        print("Loaded the best model with validation loss:", best_val_loss)

    return model, history, epoch


def test(model, test_loader, device, N_LABELS=1):
    model.eval()
    y_true = []
    y_pred = []
    y_color = []
    with torch.no_grad():
        for seqs, y in test_loader:
            seqs = seqs.float().to(device)
            labels, classes = y
            labels = torch.tensor(labels, dtype=torch.float32).to(device)

            outputs = model.abstract(seqs)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.squeeze().cpu().numpy())
            y_color.extend(classes.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_color = np.array(y_color)

    print(f"Shapes: y_true: {y_true.shape}, y_pred: {y_pred.shape}, y_color: {y_color.shape}")

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pierson_corr = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"Test MSE: {mse:.4f}, R^2: {r2:.4f}, Pearson Correlation: {pierson_corr:.4f}")

    return y_true, y_pred, y_color, mse, r2, pierson_corr