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
          lambda_rules:float = 1.0,
          Valloader:DataLoader | None = None):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CEloss = nn.CrossEntropyLoss(ignore_index=-1)
    Optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.to(device)
    # Ramp-up for rule loss weigh
    rules_ramp = torch.linspace(0.01,lambda_rules, epochs, device=device)
    for e in range(epochs):
        running_loss = 0.0
        running_ce_loss = 0.0
        running_rules_loss = 0.0
        model.train()
        for X, Y in tqdm(Trainloader):
            X, Y = X.to(device), Y.to(device)
            model.zero_grad()
            ypred = model(X) # outputs logits (B, T, sudoku_size)
            
            # Rule-enforcing loss
            # row / col / box constraints
            B, T, S = ypred.shape
            box_size = int(S**0.5)
            probs = torch.softmax(ypred, dim = -1)
            probs_board = probs.view(B, S, S, S) # (B, rows, cols, digits)
            # (B, box_row, box_col, cell_row, cell_col, digits)
            probs_boxes = probs_board.view(B, box_size, box_size, box_size, box_size, S) 
            row_sum = probs_board.sum(dim = 2) # (B, rows, digits)
            col_sum = probs_board.sum(dim = 1) # (B, cols, digits)
            box_sum = probs_boxes.sum(dim = (3,4)) # (B, box_row, box_col, digits)
            box_sum = box_sum.view(B, S, S) # (B, box, digits)
            # Calculate individual losses (row/col/box sum for each digit should be 1)
            # sum over digits (dim = 2) good: initially larger than CE but drops quickly
            # make contribution sudokusize * bigger: sum over rows / cols / boxes
            # initially huge but downweighed by ramp, 
            # by the time it's relevant, decreased substantially
            row_loss = ((row_sum - 1)**2).sum(dim = (1, 2)).mean()
            col_loss = ((col_sum - 1)**2).sum(dim = (1, 2)).mean()
            box_loss = ((box_sum - 1)**2).sum(dim = (1, 2)).mean()
            rules_loss = row_loss + col_loss + box_loss

            # For CE: Only judge on missing entries
            mask = (X == 0)
            Y[~mask] = -1 # ignored in loss calculation
            # CE expects class dimension as 2nd
            ce_loss = CEloss(ypred.transpose(1,2), Y-1)
            
            loss = ce_loss + (rules_ramp[e] * rules_loss)
            loss.backward()
            Optimizer.step()
            running_loss += loss.item()
            running_ce_loss += ce_loss.item()
            running_rules_loss += rules_loss.item()
        print(f'Epoch {e}, Average Loss: {running_loss / len(Trainloader)}\n',
              f'    CE Loss: {running_ce_loss/len(Trainloader)}\n', 
              f'    Sudoku Rules Loss: {running_rules_loss /len(Trainloader)}'
              )
        if Valloader:
            model.eval()
            running_board_acc = 0
            running_cell_acc = 0
            for xt, yt in Trainloader:
                xt, yt = xt.to(device), yt.to(device)
                ytpred = model(xt)
                ytpred = torch.argmax(ytpred, dim = -1) + 1
                running_board_acc += torch.all(ytpred == yt, dim = 1).float().mean()
                # Only judge on missing entries
                mask = (xt == 0)
                running_cell_acc += (ytpred[mask] == yt[mask]).float().mean()
            print(f'Epoch {e}, Average train accuracy - board: {running_board_acc / len(Trainloader)}',
                  f'cell: {running_cell_acc / len(Trainloader)}', 
                  flush=True)
            
            running_board_acc = 0
            running_cell_acc = 0
            for xv, yv in Valloader:
                xv, yv = xv.to(device), yv.to(device)
                yvpred = model(xv)
                yvpred = torch.argmax(yvpred, dim = -1) + 1
                running_board_acc += torch.all(yvpred == yv, dim = 1).float().mean()
                # Only judge on missing entries
                mask = (xv == 0)
                running_cell_acc += (yvpred[mask] == yv[mask]).float().mean()
            print(f'Epoch {e}, Average validation accuracy - board: {running_board_acc / len(Valloader)}', 
                  f'cell: {running_cell_acc / len(Valloader)}', 
                  flush=True)

def main(batch_size = 2**5,#5, 6, 8
         dataset_proportion:float = .003,# 0.001, .002, .01
         lambd_rules:float = 2, # 1,
         epochs:int = 50,
         lr:float = 5e-4
         ):
    ''''
    Batch size st at least 100 gradient descent steps per epoch!
    Approx once  CE Loss falls below 0.8 & per cell validation accuracy crosses 47%, 
    start having non-zero board accuracy
    '''
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

    train(SudokuModel, Trainloader=train_loader, Valloader=val_loader,
          lr= lr, epochs=epochs, lambda_rules=lambd_rules)

if __name__ == '__main__':
    main()