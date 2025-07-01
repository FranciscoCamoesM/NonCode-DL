import torch
import numpy as np

from tqdm import tqdm
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader


def gen_sampler_weights(data, n_classes):
    weights = []
    if n_classes <= 2:
        class_count = [0, 0]
        for _, label in data:
            class_count[int(label)] += 1

        for _, label in data:
            weights.append(1/class_count[int(label)])

    else:
        class_count = [0, 0, 0, 0]
        for _, label in data:
            class_count[np.argmax(label)] += 1

        for _, label in data:
            weights.append(1/class_count[np.argmax(label)])

    return torch.DoubleTensor(weights)




def count_corrects(output, target, binary=False):
    if binary:
        preds = torch.round(torch.sigmoid(output))
        return torch.sum(preds == target)
    else:
        preds = torch.argmax(output, dim=1)
        return torch.sum(preds == torch.argmax(target, dim=1))

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device='cpu', BEST_VAL=False, binary=True, early_stopping_patience=-1):
    best_model = None
    best_loss = float('inf')
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    patience = 0

    with tqdm(total=num_epochs) as pbar:
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            running_corrects = 0
            running_total = 0
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                data = data.float()
                target = target.float()

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                # nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                optimizer.step()
                train_loss += loss.item()
                running_corrects += count_corrects(output, target, binary=binary)
                running_total += len(target)
            train_losses.append(train_loss / len(train_loader))
            train_accs.append(running_corrects.item() / running_total)

            val_loss, val_acc = test(model, val_loader, criterion, device=device)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            pbar.set_postfix({'train_acc': train_accs[-1], 'val_acc': val_accs[-1], 'train_loss': train_losses[-1], 'val_loss': val_losses[-1]})
            pbar.update(1)
            pbar.set_description(f'Epoch {epoch+1}/{num_epochs}')
            pbar.refresh()

            if val_loss < best_loss:
                if BEST_VAL:
                    best_loss = val_loss
                    best_model = model.state_dict()
                    print(f'Best model found at epoch {epoch+1} with loss: {best_loss:.4f}')
                    # Save the best model
                    torch.save(best_model, 'best_model.pth')
                    patience = 0
            else:
                if early_stopping_patience > 0:
                    patience += 1
                    if patience >= early_stopping_patience:
                        print(f'Early stopping at epoch {epoch+1} with patience {patience}')
                        break
                    
            



    if BEST_VAL:
        model.load_state_dict(best_model)

    return train_losses, val_losses, train_accs, val_accs


def get_optimizer(ModelArch, N_CLASSES, LEARNING_RATE, WEIGHT_DECAY, MOM=None, OPTIM='adam'):
    model = ModelArch(num_classes=N_CLASSES)

    if N_CLASSES <= 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    if OPTIM == 'adam':
        optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    elif OPTIM == 'sgd':
        optimizer = SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, momentum=MOM)
    else:
        raise ValueError("Invalid optimizer")
    
    return model, criterion, optimizer





def update_confusion_matrix(y_true, y_pred, TP, TN, FP, FN):
    y_pred = torch.sigmoid(y_pred)
    y_pred = (y_pred > 0.5).float()

    TP += ((y_true == 1) & (y_pred == 1)).sum().item()
    TN += ((y_true == 0) & (y_pred == 0)).sum().item()
    FP += ((y_true == 0) & (y_pred == 1)).sum().item()
    FN += ((y_true == 1) & (y_pred == 0)).sum().item()

    return TP, TN, FP, FN


def test(model, test_loader, criterion, device='cpu', return_matrix=False):
    model.eval()
    test_loss = 0.0
    running_corrects = 0
    running_total = 0
    TP, TN, FP, FN = 0, 0, 0, 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()
            running_corrects += count_corrects(output, target, binary=True)
            running_total += len(target)
            TP, TN, FP, FN = update_confusion_matrix(target, output, TP, TN, FP, FN)

    test_loss /= len(test_loader)
    test_acc = running_corrects.item() / running_total
    if return_matrix:
        return test_loss, test_acc, [[TN, FP], [FN, TP]]
    else:
        return test_loss, test_acc
    
def get_metrics(confusion_matrix):
    TP, FN = confusion_matrix[0]
    FP, TN = confusion_matrix[1]

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score


def plot_confusion_matrix(confusion_matrix, save_dir=None):
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize the confusion matrix

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    if save_dir:
        plt.savefig(save_dir)
    else:
        plt.show()
    plt.close()


def save_model_specs(parameter_dict, save_dir):
    model_name = parameter_dict.get('model_name', 'model')

    with open(f"{save_dir}/{model_name}_params.txt", 'w') as f:
        for key, value in parameter_dict.items():
            f.write(f"{key}: {value}\n")



## TODO: add the function to save the model, confusion matrix, training history, and parameters all in different files
def train_and_evaluate(ModelArch, train_loader, val_loader, 
                       test_loader, N_CLASSES, ENHANCER_NAME, LABEL_TITLE, 
                       BATCH_SIZE, N_EPOCHS, LEARNING_RATE, WEIGHT_DECAY, 
                       MOM=None, OPTIM='adam', save_dir=None, best_val=False):
    """
    Train and evaluate the model.
    Args:
        ModelArch: The model architecture.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        test_loader: DataLoader for the test set.
        N_CLASSES: Number of classes.
        ENHANCER_NAME: Name of the enhancer.
        LABEL_TITLE: Title of the label.
        BATCH_SIZE: Batch size.
        N_EPOCHS: Number of epochs.
        LEARNING_RATE: Learning rate.
        WEIGHT_DECAY: Weight decay.
        MOM: Momentum (optional).
        OPTIM: Optimizer type ('adam' or 'sgd').
        save_dir: Directory to save the model and plots (optional).
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, criterion, optimizer = get_optimizer(ModelArch, N_CLASSES, LEARNING_RATE, WEIGHT_DECAY, MOM, OPTIM)
    model.to(device)
    print("Running on: ", device)


    train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device=device, BEST_VAL=best_val)
    test_loss, test_acc, confusion_matrix = test(model, test_loader, criterion, device=device, return_matrix=True)

    if save_dir:
        plot_confusion_matrix(confusion_matrix, save_dir=save_dir)
    accuracy, precision, recall, f1_score = get_metrics(confusion_matrix)
    

    model_specs = {}
    model_specs['Enhancer'] = ENHANCER_NAME
    model_specs['Label'] = LABEL_TITLE
    model_specs['Batch Size'] = BATCH_SIZE
    model_specs['Epochs'] = N_EPOCHS
    model_specs['Learning Rate'] = LEARNING_RATE
    model_specs['Weight Decay'] = WEIGHT_DECAY
    if MOM is not None:
        model_specs['Momentum'] = MOM
    model_specs['Optimizer'] = OPTIM

    
    model_specs['\nModel Test Metrics'] = ' '
    model_specs['accuracy'] = accuracy
    model_specs['precision'] = precision
    model_specs['recall'] = recall
    model_specs['f1_score'] = f1_score
        


    return train_losses, val_losses, test_loss, test_acc