# Author: @matushalak
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from sudoku_extreme import SudokuDataset
from SudokuTransformer import SudokuTransformer, AttConfig

def train(model:SudokuTransformer, 
          Trainloader:DataLoader,
          lr:float, epochs:int,
          Valloader:DataLoader | None = None):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Loss = nn.CrossEntropyLoss(ignore_index=-1)
    Optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.to(device)
    for e in range(epochs):
        running_loss = 0.0
        model.train()
        for X, Y in tqdm(Trainloader):
            X, Y = X.to(device), Y.to(device)
            model.zero_grad()
            ypred = model(X)
            # Only judge on missing entries
            mask = (X == 0)
            Y[~mask] = -1 # ignored in loss calculation
            # CE expects class dimension as 2nd
            loss = Loss(ypred.transpose(1,2), Y-1)
            loss.backward()
            Optimizer.step()
            running_loss += loss.item()
        print(f'Epoch {e}, Average Loss / batch: {running_loss / len(Trainloader)}', flush=True)
        if Valloader:
            model.eval()
            running_board_acc = 0
            running_cell_acc = 0
            for xv, yv in tqdm(Valloader):
                xv, yv = xv.to(device), yv.to(device)
                yvpred = model(xv)
                yvpred = torch.argmax(yvpred, dim = -1) + 1
                running_board_acc += torch.all(yvpred == yv, dim = 1).float().mean()
                # Only judge on missing entries
                mask = (xv == 0)
                running_cell_acc += (yvpred[mask] == yv[mask]).float().mean()
            print(f'Epoch {e}, Average validation accuracy - board: {running_board_acc / len(Valloader)}, cell: {running_cell_acc / len(Valloader)}', 
                  flush=True)

def main(batch_size = 2**5, dataset_proportion:float = 0.001):
    # Datasets
    train_set = SudokuDataset(split='train', dataset_prop=dataset_proportion)
    val_set = SudokuDataset(split='validate', dataset_prop=dataset_proportion)
    test_set = SudokuDataset(split='test')
    # Data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # Model config
    model_config = AttConfig(n_embd=128, n_head=2, resid_pdrop=0.05, attn_pdrop=0.01)

    SudokuModel = SudokuTransformer(sudoku_size=9, 
                                    n_transformer_blocks=8, 
                                    mlp_expansion=8,
                                    config=model_config)

    train(SudokuModel, Trainloader=train_loader, lr= 5e-4, epochs=30, Valloader=val_loader)

if __name__ == '__main__':
    main()