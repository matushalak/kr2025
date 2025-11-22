# Author: @matushalak
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from sudoku_extreme import SudokuDataset
from SudokuTransformer import SudokuTransformer, AttConfig

def versioned_logger(logdir:str = 'transformer_logs'):
    os.makedirs(logdir, exist_ok=True)
    # find existing version_* dirs
    existing = [d for d in os.listdir(logdir)
                if d.startswith("version_") and os.path.isdir(os.path.join(logdir, d))]

    if not existing:
        version = 0
    else:
        nums = []
        for d in existing:
            try:
                nums.append(int(d.split("_")[1]))
            except (IndexError, ValueError):
                pass
        version = max(nums) + 1 if nums else 0

    log_dir = os.path.join(logdir, f"version_{version}")
    print(f"Logging to: {log_dir}")
    return SummaryWriter(log_dir=log_dir)

def train(model:SudokuTransformer, 
          Trainloader:DataLoader,
          lr:float, epochs:int,
          lambda_rules:float = 1.0,
          Valloader:DataLoader | None = None,
          logger:SummaryWriter | None = None,
          previous_epochs:int = 0):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CEloss = nn.CrossEntropyLoss(ignore_index=-1)
    Optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.to(device)
    # Ramp-up for rule loss weigh
    if previous_epochs == 0:
        rules_ramp = torch.linspace(0.01,lambda_rules, epochs, device=device)
    else:
        rules_ramp = torch.full(size=epochs, fill_value=lambda_rules, device=device)
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
            # NOTE: actually harms training!!!
            # learning to also predict clues can only help not harm
            # mask = (X == 0)
            # Y[~mask] = -1 # ignored in loss calculation
            
            # CE expects class dimension as 2nd
            ce_loss = CEloss(ypred.transpose(1,2), Y-1)
            
            loss = ce_loss + (rules_ramp[e] * rules_loss)
            loss.backward()
            Optimizer.step()
            running_loss += loss.item()
            running_ce_loss += ce_loss.item()
            running_rules_loss += rules_loss.item()
        
        if logger:
                logger.add_scalar('train/Loss', running_loss / len(Trainloader), global_step = e + previous_epochs)
                logger.add_scalar('train/CE_loss', running_ce_loss / len(Trainloader), global_step = e + previous_epochs)
                logger.add_scalar('train/Sudoku_loss', running_rules_loss / len(Trainloader), global_step = e + previous_epochs)
        print(f'Epoch {e+previous_epochs}, Average Loss: {running_loss / len(Trainloader)}\n',
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
            if logger:
                logger.add_scalar('train/Board_Accuracy', running_board_acc / len(Trainloader), global_step = e + previous_epochs)
                logger.add_scalar('train/Cell_Accuracy', running_cell_acc / len(Trainloader), global_step = e + previous_epochs)
            print(f'Epoch {e + previous_epochs}, Average train accuracy - board: {running_board_acc / len(Trainloader)}',
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
            if logger:
                logger.add_scalar('val/Board_Accuracy', running_board_acc / len(Valloader), global_step = e + previous_epochs)
                logger.add_scalar('val/Cell_Accuracy', running_cell_acc / len(Valloader), global_step = e + previous_epochs)
            print(f'Epoch {e + previous_epochs}, Average validation accuracy - board: {running_board_acc / len(Valloader)}', 
                  f'cell: {running_cell_acc / len(Valloader)}', 
                  flush=True)

def test(model, Testloader, logger):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    running_board_acc = 0
    running_cell_acc = 0
    for xv, yv in Testloader:
        xv, yv = xv.to(device), yv.to(device)
        yvpred = model(xv)
        yvpred = torch.argmax(yvpred, dim = -1) + 1
        running_board_acc += torch.all(yvpred == yv, dim = 1).float().mean()
        # Only judge on missing entries
        mask = (xv == 0)
        running_cell_acc += (yvpred[mask] == yv[mask]).float().mean()
    if logger:
        logger.add_scalar('test/Board_Accuracy', running_board_acc / len(Testloader))
        logger.add_scalar('test/Cell_Accuracy', running_cell_acc / len(Testloader))
    print(f'Average test accuracy - board: {running_board_acc / len(Testloader)}', 
          f'cell: {running_cell_acc / len(Testloader)}', flush=True)

    

def curriculum(guess_stages:list[int] = [30, 35, 40, 45, 50, None],
               epochs_per_stage:int = 10,
               batch_size = 2**6,#5, 6, 8
               dataset_proportion:float = .01,# 0.001, .002, .01
               lambd_rules:float = 1, # 1,
               lr:float = 5e-4,):
    # logger setup
    # Logger:SummaryWriter = versioned_logger()
    Logger = None

    # Model config
    model_config = AttConfig(n_embd=128, n_head=4, 
                             resid_pdrop=0.01, attn_pdrop=0.0
                             )
    _SudokuModel_ = SudokuTransformer(sudoku_size=9, 
                                      n_transformer_blocks=8, 
                                      mlp_expansion=8,
                                      config=model_config)
    cum_epochs = 0
    # Loop through stages of curriculum
    for guess in guess_stages:
        print(guess, ' empty cells to guess.')
        run_(batch_size=batch_size, 
             dataset_proportion=dataset_proportion,
             lambd_rules = lambd_rules, lr = lr,
             epochs=epochs_per_stage, 
             NGUESS=guess,
             SudokuModel=_SudokuModel_,
             logger=Logger,
             prev_epochs=cum_epochs)
        cum_epochs += epochs_per_stage
    
    # test_set = SudokuDataset(split='test')
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # test(_SudokuModel_, test_loader, Logger)
    # Logger.close()

def run_(batch_size = 2**6,#5, 6, 8
         dataset_proportion:float = .01,# 0.001, .002, .01
         lambd_rules:float = 1, # 1,
         epochs:int = 10, # 100
         lr:float = 5e-4,
         NGUESS:int = None,
         SudokuModel:SudokuTransformer = None,
         logger:SummaryWriter = None,
         prev_epochs:int = 0
         ):
    ''''
    Batch size st at least 100 gradient descent steps per epoch!
    Approx once  CE Loss falls below 0.8 & per cell validation accuracy crosses 47%, 
    start having non-zero board accuracy
    '''
    # Datasets
    train_set = SudokuDataset(split='train', dataset_prop=dataset_proportion,
                              n_guess=NGUESS)
    val_set = SudokuDataset(split='validate', dataset_prop=dataset_proportion,
                            n_guess=NGUESS)
    
    # Data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    
    # Model config
    model_config = AttConfig(n_embd=128, n_head=4, 
                             resid_pdrop=0.01, attn_pdrop=0.0
                             )
    if SudokuModel is None:
        SudokuModel = SudokuTransformer(sudoku_size=9, 
                                        n_transformer_blocks=8, 
                                        mlp_expansion=8,
                                        config=model_config)

    train(SudokuModel, Trainloader=train_loader, Valloader=val_loader,
          lr= lr, epochs=epochs, lambda_rules=lambd_rules,
          logger=logger, previous_epochs=prev_epochs)
    

if __name__ == '__main__':
    curriculum(dataset_proportion=0.01,
               batch_size=2**6,
               epochs_per_stage=5)