import torch
from tqdm import tqdm
import numpy as np

@torch.no_grad()
def evaluate_ppl(model, loader, tokenizer):
    total_loss = 0
    total_count = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.clone()
            if batch.shape[1] <= 1: continue
            input_ids = batch.to(model.device)
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            if tokenizer.pad_token_id is not None:
                count = input_ids.ne(tokenizer.pad_token_id).ne(-100).sum().item()
            else:
                count = input_ids.ne(-100).sum().item()
            total_loss += loss.item() * count
            total_count += count
            
    return np.exp(total_loss / total_count)

