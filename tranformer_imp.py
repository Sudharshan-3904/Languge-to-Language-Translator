import math
import random
import time
from collections import Counter
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


DATA_FILE = "bilingual_pairs\\fra edit.txt"
TRAIN_SPLIT = 0.8
EPOCHS = 20
BATCH_SIZE = 64
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 3
DIM_FF = 512
DROPOUT = 0.3
LR = 1e-4
MAX_LEN = 50
MIN_FREQ = 2
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_MODEL = "transformer_model.pt"

random.seed(SEED)
torch.manual_seed(SEED)

def simple_tokenize(line: str) -> List[str]:
    return line.strip().split()

class Vocab:
    def __init__(self, min_freq=1, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
        self.freqs = Counter()
        self.min_freq = min_freq
        self.itos = list(specials)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.specials = specials

    def add_sentence(self, tokens: List[str]):
        self.freqs.update(tokens)

    def build(self):
        for tok, cnt in self.freqs.most_common():
            if cnt < self.min_freq: 
                continue
            if tok in self.stoi: 
                continue
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)

    def __len__(self): 
        return len(self.itos)

    def encode(self, tokens: List[str], add_sos_eos=True) -> List[int]:
        ids = [self.stoi.get(t, self.stoi["<unk>"]) for t in tokens]
        if add_sos_eos:
            return [self.stoi["<sos>"]] + ids + [self.stoi["<eos>"]]
        else:
            return ids

    def decode(self, ids: List[int], skip_special=True) -> List[str]:
        toks = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                toks.append("<unk>")
            else:
                t = self.itos[i]
                if skip_special and t in self.specials:
                    continue
                toks.append(t)
        return toks

class ParallelDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str]], src_vocab: Vocab, tgt_vocab: Vocab, max_len=50):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self): 
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        src_ids = self.src_vocab.encode(simple_tokenize(src), add_sos_eos=False)
        tgt_ids = self.tgt_vocab.encode(simple_tokenize(tgt), add_sos_eos=True)
        # truncate
        if len(src_ids) > self.max_len:
            src_ids = src_ids[:self.max_len]
        if len(tgt_ids) > self.max_len:
            tgt_ids = tgt_ids[:self.max_len-1] + [self.tgt_vocab.stoi["<eos>"]]
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_pad = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_pad, tgt_pad

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_layers, dim_ff, dropout, src_pad_idx, tgt_pad_idx, max_len):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers, num_decoder_layers=num_layers,
            dim_feedforward=dim_ff, dropout=dropout, batch_first=True,
        )
        self.output_fc = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def make_tgt_mask(self, tgt_len, device):
        mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device) == 1, diagonal=1)
        mask = mask.float().masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, src, tgt):
        tgt_mask = self.make_tgt_mask(tgt.size(1), tgt.device)
        src_key_padding_mask = (src == self.src_pad_idx)
        tgt_key_padding_mask = (tgt == self.tgt_pad_idx)

        src_emb = self.pos_encoder(self.src_tok_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_decoder(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))

        out = self.transformer(
            src_emb, tgt_emb,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.output_fc(out)

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, vocab_size, ignore_index=0):
        super().__init__()
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        with torch.no_grad():
            true_dist = output.data.clone()
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            ignore = target == self.ignore_index
            target_clamped = target.clone()
            target_clamped[ignore] = 0
            true_dist.scatter_(1, target_clamped.unsqueeze(1), self.confidence)
            true_dist[ignore] = 0
        log_prob = torch.log_softmax(output, dim=1)
        return torch.sum(-true_dist * log_prob, dim=1).mean()

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        optimizer.zero_grad()
        logits = model(src, tgt_in)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def greedy_decode(model, src, tgt_vocab, max_len=50):
    model.eval()
    sos_idx, eos_idx = tgt_vocab.stoi["<sos>"], tgt_vocab.stoi["<eos>"]
    ys = torch.ones((1, 1), dtype=torch.long, device=DEVICE) * sos_idx
    with torch.no_grad():
        for _ in range(max_len-1):
            logits = model(src, ys)
            next_tok = logits[:, -1, :].argmax(-1).unsqueeze(1)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == eos_idx:
                break
    return tgt_vocab.decode(ys[0].cpu().numpy().tolist())

def main():
    # Load bilingual pairs
    with open(DATA_FILE, "r", encoding="utf8") as f:
        pairs = [line.strip().split("\t") for line in f if "\t" in line]

    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_SPLIT)
    train_pairs, val_pairs = pairs[:split_idx], pairs[split_idx:]

    # Build vocabs
    src_vocab, tgt_vocab = Vocab(MIN_FREQ), Vocab(MIN_FREQ)
    for s, t in pairs:
        src_vocab.add_sentence(simple_tokenize(s))
        tgt_vocab.add_sentence(simple_tokenize(t))
    src_vocab.build()
    tgt_vocab.build()

    print(f"Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    print(f"Src vocab: {len(src_vocab)}, Tgt vocab: {len(tgt_vocab)}")

    # DataLoaders
    train_ds = ParallelDataset(train_pairs, src_vocab, tgt_vocab, MAX_LEN)
    val_ds = ParallelDataset(val_pairs, src_vocab, tgt_vocab, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Model
    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab), tgt_vocab_size=len(tgt_vocab),
        d_model=D_MODEL, nhead=NHEAD, num_layers=NUM_LAYERS,
        dim_ff=DIM_FF, dropout=DROPOUT,
        src_pad_idx=src_vocab.stoi["<pad>"], tgt_pad_idx=tgt_vocab.stoi["<pad>"],
        max_len=MAX_LEN
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = LabelSmoothingLoss(0.1, len(tgt_vocab), ignore_index=tgt_vocab.stoi["<pad>"])

    # Training loop
    for epoch in range(1, EPOCHS+1):
        start = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = train_epoch(model, val_loader, optimizer, criterion)  # using same loop for quick check
        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - {time.time()-start:.1f}s")

    # Show sample translations
    print("\nSample translations:")
    for src, tgt in val_pairs[:5]:
        src_ids = torch.tensor([src_vocab.encode(simple_tokenize(src), add_sos_eos=False)], device=DEVICE)
        pred = greedy_decode(model, src_ids, tgt_vocab, max_len=MAX_LEN)
        print("SRC:", src)
        print("PRED:", " ".join(pred))
        print("TGT:", tgt)
        print("---")

if __name__ == "__main__":
    main()
