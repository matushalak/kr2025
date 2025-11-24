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
          previous_epochs:int = 0,
          Optimizer:optim.Optimizer | None = None):
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CEloss = nn.CrossEntropyLoss(ignore_index=-1)
    if Optimizer is None:
        print(f'Initiating new AdamW optimizer with lr = {lr}')
        Optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    model.to(device)
    # Ramp-up for rule loss weigh
    if previous_epochs == 0:
        rules_ramp = torch.linspace(1e-8,lambda_rules, epochs, device=device)
    else:
        rules_ramp = torch.full(size = (epochs, ), fill_value=lambda_rules, device=device)
    
    for e in range(epochs):
        running_loss = 0.0
        running_ce_loss = 0.0
        running_rules_loss = 0.0
        model.train()
        for X, Y in tqdm(Trainloader):
            X, Y = X.to(device), Y.to(device)
            
            Optimizer.zero_grad()
            ypred = model(X) # outputs logits (B, T, sudoku_size)
            
            # Rule-enforcing loss
            empties = (X == 0)
            clues = ~empties
            # row / col / box constraints
            B, T, S = ypred.shape
            box_size = int(S**0.5)
            probs = torch.softmax(ypred, dim = -1)
            # Givens / clues should are 'one-hot' in softmax
            clues_one_hot = nn.functional.one_hot(Y-1, num_classes=S).float()
            # effective probabilities (after taking care of clues)
            probs_eff = (probs * empties[..., None]) + (clues_one_hot * clues[..., None])
            probs_board = probs_eff.view(B, S, S, S) # (B, rows, cols, digits)
            # (B, box_row, box_col, cell_row, cell_col, digits)
            probs_boxes = probs_board.view(B, box_size, box_size, box_size, box_size, S)
            # NOTE: Using sum of probabilities instead of square probabilities 
            # was converging to uniform dist! at harder curricula 40+ 
            row_mass = (probs_board ** 2).sum(dim = 2) # (B, rows, digits)
            col_mass = (probs_board ** 2).sum(dim = 1) # (B, cols, digits)
            box_mass = (probs_boxes ** 2).sum(dim = (3,4)) # (B, box_row, box_col, digits)
            box_mass = box_mass.view(B, S, S) # (B, box, digits)
            # Calculate individual losses (row/col/box sum for each digit should be 1)
            # sum over digits (dim = 2) good: initially larger than CE but drops quickly
            # make contribution sudokusize * bigger: sum over rows / cols / boxes
            # initially huge but downweighed by ramp, 
            # by the time it's relevant, decreased substantially
            row_loss = ((row_mass - 1)**2).sum(dim = (1, 2)).mean()
            col_loss = ((col_mass - 1)**2).sum(dim = (1, 2)).mean()
            box_loss = ((box_mass - 1)**2).sum(dim = (1, 2)).mean()
            rules_loss = row_loss + col_loss + box_loss

            # need to match 0-8 indices instead of 1-9 digits
            Y -= 1
            # For CE: Only judge on missing entries
            # NOTE: maybe actually harms training!!!
            # learning to also predict clues can only help not harm
            # Y[clues] = -1 # ignored in loss calculation
            
            # CE expects class dimension as 2nd
            ce_loss = CEloss(ypred.transpose(1,2), Y)
            
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


def run_(batch_size = 2**6,#5, 6, 8
         dataset_proportion:float = .01,# 0.001, .002, .01
         lambd_rules:float = 1, # 1,
         epochs:int = 10, # 100
         lr:float = 5e-4,
         NGUESS:int = None,
         SudokuModel:SudokuTransformer = None,
         logger:SummaryWriter = None,
         prev_epochs:int = 0,
         Optimizer:optim.Optimizer | None = None
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
    
    
    if SudokuModel is None:
        model_config = AttConfig(n_embd=128, n_head=4, 
                                resid_pdrop=0.01, attn_pdrop=0.0
                                )
        SudokuModel = SudokuTransformer(sudoku_size=9, 
                                        n_transformer_blocks=8, 
                                        mlp_expansion=8,
                                        config=model_config)

    train(SudokuModel, Trainloader=train_loader, Valloader=val_loader,
          lr= lr, epochs=epochs, lambda_rules=lambd_rules,
          logger=logger, previous_epochs=prev_epochs,
          Optimizer=Optimizer)
    

def curriculum(guess_stages:list[int] = [30, 35, 40, 45, 50, None],
               epochs_per_stage:int|list[int] = 10,
               batch_size = 2**6,#5, 6, 8
               dataset_proportion:float = .01,# 0.001, .002, .01
               lambd_rules:float|list[float] = 1.0, # 1,
               learning_rate:float|list[float] = 5e-4,
               recurrence:int = 1):
    # logger setup
    Logger:SummaryWriter = versioned_logger()
    # Logger = None

    # Model config
    model_config = AttConfig(n_embd=150, n_head=3, 
                             resid_pdrop=0.01, attn_pdrop=0.005,
                             recurrence=recurrence
                             )
    _SudokuModel_ = SudokuTransformer(sudoku_size=9, 
                                      n_transformer_blocks=5, 
                                      mlp_expansion=4,
                                      config=model_config)
    cum_epochs = 0
    if isinstance(epochs_per_stage, list):
        assert len(epochs_per_stage) == len(guess_stages)
    if isinstance(lambd_rules, list):
        assert len(epochs_per_stage) == len(guess_stages)
    if isinstance(learning_rate, list):
            assert len(learning_rate) == len(guess_stages)
            Optimizer = None
    else:
        Optimizer = optim.AdamW(_SudokuModel_.parameters(), 
                                lr=learning_rate, betas=(0.9, 0.95), 
                                weight_decay=0.01)
    # Loop through stages of curriculum
    for ig, guess in enumerate(guess_stages):
        if isinstance(epochs_per_stage, list):
            eps = epochs_per_stage[ig]
        else:
            eps = epochs_per_stage
        if isinstance(lambd_rules, list):
            lambd_ = lambd_rules[ig]
        else:
            lambd_ = lambd_rules
        if isinstance(learning_rate, list):
            lr = learning_rate[ig]
        else:
            lr =  learning_rate
        if isinstance(batch_size, list):
            bs = batch_size[ig]
        else:
            bs = batch_size
        print(f'{guess} empty cells to guess | ',
              f'Training for {eps} epochs | ',
              f'with {lambd_} weight on sudoku rules loss')
        run_(batch_size=bs, 
             dataset_proportion=dataset_proportion,
             lambd_rules = lambd_, lr = lr,
             epochs=eps, 
             NGUESS=guess,
             SudokuModel=_SudokuModel_,
             logger=Logger,
             prev_epochs=cum_epochs,
             Optimizer = Optimizer)
        cum_epochs += eps
    
    # test_set = SudokuDataset(split='test')
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    # test(_SudokuModel_, test_loader, Logger)
    # Logger.close()


if __name__ == '__main__':
    curriculum(dataset_proportion=0.03,
               batch_size=2**7,
               # starting at 30, can quickly reach 80% board accuract
               # starting at 35 much harder
               guess_stages=[50],
               epochs_per_stage=[10],
               lambd_rules = 1e-3,
               learning_rate= 5e-4,
               recurrence=4
               )
# all below dataset 0.03
# 3 x Reccurrence
# v23: added recurrence starting from 36 empties
# v24: added recurrence starting from 40 empties -> 83% full-board accuracy after 1 epoch
# v25: recurrent starting at full puzzle at once (too hard, stuck around 2%)
# v26: recurrent from 50 empties - up to 42% in 10 epochs
# v27: recurrent curriculum 40-45-50 (5-5-10 epochs) (lr 5e-4) 
#   - not better than starting from 50 straight away
# 9 x recurrence
# v28: recurrent from 50, more recurrence: random (epoch 0&1)
# 2 x recurrence
# v29: recurrent from 50, more recurrence: meh
# 4x recurrence
# v30: recurrent from 50, more recurrence -> 52.8% BEST
# 5x recurrence
# v31: recurrent from 50, more recurrence: random(epoch 0&1)

# v32: 4x recurrent from 50, dataset 0.06 - not big difference from 0.06, just much longer to train