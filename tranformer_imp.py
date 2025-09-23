#!/usr/bin/env python3
"""
train_transformer.py

Small Transformer seq2seq for language-to-language translation (short -> medium sentences).
Single-file end-to-end script:
- Build vocab from parallel text files (src.txt, tgt.txt)
- Train a small Transformer
- Validate and print simple BLEU-1/2 statistics
- Greedy decode inference

Usage:
  python train_transformer.py --src train.src --tgt train.tgt --val_src val.src --val_tgt val.tgt

If you already have tokenized integer datasets, adapt the DataLoader section accordingly.
"""

import math
import argparse
import random
import time
from collections import Counter, defaultdict
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# -----------------------
# Config and arguments
# -----------------------
parser = argparse.ArgumentParser()
parser.add_argument("--src", type=str, default="train.src", help="source training file (one sentence per line)")
parser.add_argument("--tgt", type=str, default="train.tgt", help="target training file (one sentence per line)")
parser.add_argument("--val_src", type=str, default="val.src", help="validation source file")
parser.add_argument("--val_tgt", type=str, default="val.tgt", help="validation target file")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--d_model", type=int, default=128)   # model dimension
parser.add_argument("--nhead", type=int, default=4)
parser.add_argument("--num_encoder_layers", type=int, default=3)
parser.add_argument("--num_decoder_layers", type=int, default=3)
parser.add_argument("--dim_feedforward", type=int, default=512)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--max_len", type=int, default=50)
parser.add_argument("--min_freq", type=int, default=2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--save", type=str, default="transformer_model.pt")
args = parser.parse_args([]) if False else parser.parse_args([])  # in notebooks this avoids CLI parsing; else pass real args
# If running from command line, comment the above line and use the real command-line args (uncomment next)
# args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

DEVICE = torch.device(args.device)

# -----------------------
# Utilities: tokenization & vocab
# -----------------------
def simple_tokenize(line: str) -> List[str]:
    # Basic whitespace tokenizer. Replace with better tokenizer if you want (e.g. spaCy, Moses)
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

# -----------------------
# Load parallel corpus and build vocab
# -----------------------
def load_parallel(src_file: str, tgt_file: str):
    src_lines, tgt_lines = [], []
    with open(src_file, "r", encoding="utf8") as fsrc, open(tgt_file, "r", encoding="utf8") as ftgt:
        for sline, tline in zip(fsrc, ftgt):
            sline = sline.strip()
            tline = tline.strip()
            if not sline or not tline:
                continue
            src_lines.append(sline)
            tgt_lines.append(tline)
    return src_lines, tgt_lines

def build_vocabs(src_lines, tgt_lines, min_freq=2):
    src_vocab = Vocab(min_freq=min_freq)
    tgt_vocab = Vocab(min_freq=min_freq)
    for s in src_lines:
        src_vocab.add_sentence(simple_tokenize(s))
    for t in tgt_lines:
        tgt_vocab.add_sentence(simple_tokenize(t))
    src_vocab.build()
    tgt_vocab.build()
    return src_vocab, tgt_vocab

# -----------------------
# Dataset & collate
# -----------------------
class ParallelDataset(Dataset):
    def __init__(self, src_lines: List[str], tgt_lines: List[str], src_vocab: Vocab, tgt_vocab: Vocab, max_len=50):
        assert len(src_lines) == len(tgt_lines)
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_tokens = simple_tokenize(self.src_lines[idx])
        tgt_tokens = simple_tokenize(self.tgt_lines[idx])
        src_ids = self.src_vocab.encode(src_tokens, add_sos_eos=False)  # we can omit sos for src
        tgt_ids = self.tgt_vocab.encode(tgt_tokens, add_sos_eos=True)
        # truncate if too long (keep eos)
        if len(src_ids) > self.max_len:
            src_ids = src_ids[:self.max_len]
        if len(tgt_ids) > self.max_len:
            tgt_ids = tgt_ids[:self.max_len-1] + [self.tgt_vocab.stoi["<eos>"]]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(x) for x in src_batch]
    tgt_lens = [len(x) for x in tgt_batch]
    src_pad = nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_pad = nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_pad, tgt_pad, torch.tensor(src_lens), torch.tensor(tgt_lens)

# -----------------------
# Positional Encoding
# -----------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

# -----------------------
# Transformer Seq2Seq Model
# -----------------------
class TransformerSeq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=512,
        dropout=0.2,
        src_pad_idx=0,
        tgt_pad_idx=0,
        max_len=500
    ):
        super().__init__()
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.output_fc = nn.Linear(d_model, tgt_vocab_size)
        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def make_src_key_padding_mask(self, src):
        # src: (batch, src_len)
        return (src == self.src_pad_idx)

    def make_tgt_key_padding_mask(self, tgt):
        return (tgt == self.tgt_pad_idx)

    def make_tgt_mask(self, tgt_len):
        # square subsequent mask (causal) for target
        mask = torch.triu(torch.ones((tgt_len, tgt_len), device=next(self.parameters()).device) == 1, diagonal=1)
        mask = mask.float().masked_fill(mask == 1, float("-inf"))
        return mask

    def forward(self, src, tgt):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)  including <sos> at position 0
        returns: logits (batch, tgt_len, vocab)
        """
        src_mask = None
        tgt_mask = self.make_tgt_mask(tgt.size(1)).to(tgt.device)  # (tgt_len, tgt_len)
        src_key_padding_mask = self.make_src_key_padding_mask(src)  # (batch, src_len)
        tgt_key_padding_mask = self.make_tgt_key_padding_mask(tgt)  # (batch, tgt_len)

        src_emb = self.src_tok_emb(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)

        tgt_emb = self.tgt_tok_emb(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)

        out = self.transformer(
            src_emb,
            tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=None,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )  # (batch, tgt_len, d_model)

        logits = self.output_fc(out)
        return logits

# -----------------------
# Label smoothing loss
# -----------------------
class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=0):
        super().__init__()
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.vocab_size = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output: (N, C) logits
        target: (N,)
        """
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

# -----------------------
# Evaluation: simple BLEU-1/2 (precision) without external libs
# -----------------------
def compute_corpus_bleu(preds: List[List[str]], refs: List[List[str]]):
    # preds and refs are lists of token lists (strings). We compute unigram and bigram precision averaged
    def ngram_counts(tokens, n):
        c = Counter()
        for i in range(len(tokens)-n+1):
            c[tuple(tokens[i:i+n])] += 1
        return c

    total_pred_unigrams = 0
    matched_unigrams = 0
    total_pred_bigrams = 0
    matched_bigrams = 0

    for p, r in zip(preds, refs):
        # unigram
        p1 = Counter(p)
        r1 = Counter(r)
        for tok, cnt in p1.items():
            matched_unigrams += min(cnt, r1.get(tok, 0))
        total_pred_unigrams += max(len(p), 1)

        # bigram
        p2 = ngram_counts(p, 2)
        r2 = ngram_counts(r, 2)
        for ng, cnt in p2.items():
            matched_bigrams += min(cnt, r2.get(ng, 0))
        total_pred_bigrams += max(sum(p2.values()), 1)

    bleu1 = matched_unigrams / total_pred_unigrams
    bleu2 = matched_bigrams / total_pred_bigrams
    return bleu1, bleu2

# -----------------------
# Greedy decode
# -----------------------
def greedy_decode(model: TransformerSeq2Seq, src: torch.Tensor, src_vocab: Vocab, tgt_vocab: Vocab, max_len=50):
    """
    src: (batch, src_len) input ids
    returns: list of token lists (strings)
    """
    model.eval()
    batch_size = src.size(0)
    src = src.to(next(model.parameters()).device)
    # prepare initial target with <sos>
    sos_idx = tgt_vocab.stoi["<sos>"]
    eos_idx = tgt_vocab.stoi["<eos>"]
    ys = torch.ones((batch_size, 1), dtype=torch.long, device=src.device) * sos_idx

    with torch.no_grad():
        for i in range(max_len - 1):
            logits = model(src, ys)  # (batch, tgt_len, vocab)
            next_tok_logits = logits[:, -1, :]  # (batch, vocab)
            next_tok = torch.argmax(next_tok_logits, dim=-1).unsqueeze(1)  # (batch, 1)
            ys = torch.cat([ys, next_tok], dim=1)
            if (next_tok == eos_idx).all():
                break
    results = []
    for seq in ys.cpu().numpy():
        # drop leading <sos> and tokens after <eos>
        toks = []
        for id_ in seq[1:]:
            if id_ == eos_idx:
                break
            if id_ == tgt_vocab.stoi["<pad>"]:
                continue
            toks.append(tgt_vocab.itos[id_])
        results.append(toks)
    return results

# -----------------------
# Training & validation
# -----------------------
def train_epoch(model, dataloader, optimizer, criterion, tgt_pad_idx):
    model.train()
    total_loss = 0.0
    for src, tgt, src_lens, tgt_lens in dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)
        # decoder input = all except last token
        tgt_input = tgt[:, :-1]
        # target for loss = all except first token
        tgt_out = tgt[:, 1:].contiguous()
        optimizer.zero_grad()
        logits = model(src, tgt_input)  # (batch, tgt_len_in, vocab)
        logits_flat = logits.view(-1, logits.size(-1))
        tgt_flat = tgt_out.view(-1)
        loss = criterion(logits_flat, tgt_flat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, src_vocab, tgt_vocab):
    model.eval()
    total_loss = 0.0
    preds = []
    refs = []
    with torch.no_grad():
        for src, tgt, src_lens, tgt_lens in dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_out = tgt[:, 1:].contiguous()
            logits = model(src, tgt_input)
            logits_flat = logits.view(-1, logits.size(-1))
            tgt_flat = tgt_out.view(-1)
            loss = criterion(logits_flat, tgt_flat)
            total_loss += loss.item()

            # greedy decode for BLEU
            batch_preds = greedy_decode(model, src.cpu(), src_vocab, tgt_vocab, max_len=args.max_len)
            for p in batch_preds:
                preds.append(p)
            for row in tgt.cpu().numpy():
                # decode target (skip <sos>, stop at <eos>)
                decoded = []
                for id_ in row[1:]:
                    if id_ == tgt_vocab.stoi["<eos>"]:
                        break
                    if id_ == tgt_vocab.stoi["<pad>"]:
                        continue
                    decoded.append(tgt_vocab.itos[id_])
                refs.append(decoded)

    avg_loss = total_loss / len(dataloader)
    bleu1, bleu2 = compute_corpus_bleu(preds, refs)
    return avg_loss, bleu1, bleu2

# -----------------------
# Entry point
# -----------------------
def main():
    # --- USER: point to your train/val files here or use args from CLI ---
    # each line corresponds to one sentence; parallel files aligned line-by-line
    try:
        train_src_lines, train_tgt_lines = load_parallel(args.src, args.tgt)
        val_src_lines, val_tgt_lines = load_parallel(args.val_src, args.val_tgt)
    except Exception as e:
        print("Error loading files. Make sure files exist and are parallel text files. Exception:", e)
        return

    print(f"Train pairs: {len(train_src_lines)} - Val pairs: {len(val_src_lines)}")

    print("Building vocabularies (min_freq=%d)..." % args.min_freq)
    src_vocab, tgt_vocab = build_vocabs(train_src_lines + val_src_lines, train_tgt_lines + val_tgt_lines, min_freq=args.min_freq)
    print("Source vocab size:", len(src_vocab))
    print("Target vocab size:", len(tgt_vocab))

    train_dataset = ParallelDataset(train_src_lines, train_tgt_lines, src_vocab, tgt_vocab, max_len=args.max_len)
    val_dataset = ParallelDataset(val_src_lines, val_tgt_lines, src_vocab, tgt_vocab, max_len=args.max_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = TransformerSeq2Seq(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        src_pad_idx=src_vocab.stoi["<pad>"],
        tgt_pad_idx=tgt_vocab.stoi["<pad>"],
        max_len=args.max_len,
    ).to(DEVICE)

    print(model)

    # criterion with label smoothing
    label_smoothing = 0.1
    criterion = LabelSmoothingLoss(label_smoothing, tgt_vocab_size=len(tgt_vocab), ignore_index=tgt_vocab.stoi["<pad>"])

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optional scheduler (reduce lr on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.7, patience=2, verbose=True)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, tgt_vocab.stoi["<pad>"])
        val_loss, bleu1, bleu2 = evaluate(model, val_loader, criterion, src_vocab, tgt_vocab)
        scheduler.step(val_loss)
        elapsed = time.time() - start_time
        print(f"Epoch {epoch}/{args.epochs} - {elapsed:.1f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - BLEU1: {bleu1:.4f} BLEU2: {bleu2:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "src_vocab": src_vocab.itos,
                "tgt_vocab": tgt_vocab.itos,
                "args": vars(args),
            }, args.save)
            print("Saved best model to", args.save)

    print("Training finished. Best val loss:", best_val_loss)
    # sample decode on a few val examples
    print("\nSample translations (val):")
    model.eval()
    for i in range(min(5, len(val_src_lines))):
        src_line = val_src_lines[i]
        tokens = simple_tokenize(src_line)
        src_ids = src_vocab.encode(tokens, add_sos_eos=False)
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0)
        pred_tokens = greedy_decode(model, src_tensor, src_vocab, tgt_vocab, max_len=args.max_len)[0]
        print("SRC:", src_line)
        print("PRED:", " ".join(pred_tokens))
        print("TGT:", val_tgt_lines[i])
        print("---")

if __name__ == "__main__":
    main()
