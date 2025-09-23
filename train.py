import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
import pickle


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)


def read_text(filename):
	with open(filename, mode='rt', encoding='utf-8') as file:
		return file.read()


def to_lines(text):
	sents = text.strip().split('\n')
	sents = [i.split('\t') for i in sents]
	return sents

# Read source and target language pairs


MAX_PAIRS = 5000
data = read_text('bilingual_pairs/fra edit.txt')
lines = to_lines(data)[:MAX_PAIRS]
src_sentences = [line[0].lower() for line in lines if len(line) > 1]
tgt_sentences = [line[1].lower() for line in lines if len(line) > 1]
min_len = min(len(src_sentences), len(tgt_sentences))
src_sentences = src_sentences[:min_len]
tgt_sentences = tgt_sentences[:min_len]

# Tokenization

# Simple tokenizer and vocab builder

def build_vocab(sentences):
	words = [word for sent in sentences for word in sent.split()]
	counter = Counter(words)
	vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common())}
	vocab['<PAD>'] = 0
	vocab['<UNK>'] = 1
	return vocab


def encode_sentence(sentence, vocab, length):
	tokens = [vocab.get(word, vocab['<UNK>']) for word in sentence.split()]
	if len(tokens) < length:
		tokens += [vocab['<PAD>']] * (length - len(tokens))
	else:
		tokens = tokens[:length]
	return tokens

src_vocab = build_vocab(src_sentences)
tgt_vocab = build_vocab(tgt_sentences)
src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)
src_length = max(len(s.split()) for s in src_sentences)
tgt_length = max(len(s.split()) for s in tgt_sentences)

print(f'Source vocab size: {src_vocab_size}, Target vocab size: {tgt_vocab_size}')
print(f'Source max length: {src_length}, Target max length: {tgt_length}')


def encode_sequences(sentences, vocab, length):
	return np.array([encode_sentence(s, vocab, length) for s in sentences])

# Train/test split
train_src, test_src, train_tgt, test_tgt = train_test_split(
	src_sentences, tgt_sentences, test_size=0.2, random_state=42)


trainX = encode_sequences(train_src, src_vocab, src_length)
trainY = encode_sequences(train_tgt, tgt_vocab, tgt_length)
testX = encode_sequences(test_src, src_vocab, src_length)
testY = encode_sequences(test_tgt, tgt_vocab, tgt_length)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim=256, hid_dim=512, n_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim, hid_dim, num_layers=n_layers, bidirectional=True, batch_first=True)
        self.fc_hidden = nn.Linear(hid_dim * 2, hid_dim)
        self.fc_cell = nn.Linear(hid_dim * 2, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_lengths=None):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))  # [batch, src_len, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded) 
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch, hid_dim*2]
        cell_cat   = torch.cat((cell[-2], cell[-1]), dim=1)
        hidden = torch.tanh(self.fc_hidden(hidden_cat)).unsqueeze(0)  # [1, batch, hid_dim]
        cell   = torch.tanh(self.fc_cell(cell_cat)).unsqueeze(0)
        return outputs, (hidden, cell)
        # outputs are used for attention

class LuongAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        # always project encoder outputs down to decoder hidden size
        self.proj = nn.Linear(enc_hid_dim, dec_hid_dim, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden: [batch, dec_hid_dim]
        # encoder_outputs: [batch, src_len, enc_hid_dim]

        # project encoder outputs to decoder hidden size
        enc = self.proj(encoder_outputs)  # [batch, src_len, dec_hid_dim]

        dec = decoder_hidden.unsqueeze(1)  # [batch, 1, dec_hid_dim]
        scores = torch.bmm(dec, enc.transpose(1, 2)).squeeze(1)  # [batch, src_len]

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=1)  # [batch, src_len]
        context = torch.bmm(attn.unsqueeze(1), enc).squeeze(1)  # [batch, dec_hid_dim]

        return context, attn

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim=256, enc_hid_dim=1024, dec_hid_dim=512, n_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(emb_dim + dec_hid_dim, dec_hid_dim, num_layers=n_layers, batch_first=True)
        self.attention = LuongAttention(enc_hid_dim, dec_hid_dim)
        self.fc_out = nn.Linear(dec_hid_dim + dec_hid_dim + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, input_token, prev_hidden, prev_cell, encoder_outputs, mask=None):
        # input_token: [batch] (one timestep)
        input_token = input_token.unsqueeze(1)  # [batch,1]
        embedded = self.dropout(self.embedding(input_token))  # [batch,1,emb_dim]
        # compute attention context using prev_hidden
        dec_hidden = prev_hidden.squeeze(0)  # [batch, dec_hid_dim]
        context, attn = self.attention(dec_hidden, encoder_outputs, mask=mask)  # [batch, enc_hid_dim]
        # combine embedded and context -> rnn input
        rnn_input = torch.cat((embedded, context.unsqueeze(1)), dim=2)  # [batch,1, emb_dim+enc_hid_dim]
        output, (hidden, cell) = self.rnn(rnn_input, (prev_hidden, prev_cell))
        # output: [batch,1,dec_hid_dim]
        output = output.squeeze(1)  # [batch, dec_hid_dim]
        # Final prediction combines output, context and embedded
        output_cat = torch.cat((output, context, embedded.squeeze(1)), dim=1)  # [batch, dec+enc+emb]
        prediction = self.fc_out(output_cat)  # [batch, output_dim]
        return prediction, hidden, cell, attn

class AttnSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx

    def create_mask(self, src):
        # src: [batch, src_len]
        mask = (src != self.pad_idx).to(device)  # [batch, src_len]
        return mask

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch, src_len], trg: [batch, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.embedding.num_embeddings

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)
        encoder_outputs, (hidden, cell) = self.encoder(src)
        mask = self.create_mask(src)

        # first input to the decoder is <sos> token — assume trg[:,0] is <sos>
        input_tok = trg[:, 0]

        for t in range(1, trg_len):
            pred, hidden, cell, attn = self.decoder(input_tok, hidden, cell, encoder_outputs, mask=mask)
            outputs[:, t, :] = pred
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = pred.argmax(1)
            input_tok = trg[:, t] if teacher_force else top1

        return outputs

    def translate(self, src_sentence_tensor, trg_sos_idx, max_len=50):
        # Greedy decode (for inference)
        self.eval()
        with torch.no_grad():
            encoder_outputs, (hidden, cell) = self.encoder(src_sentence_tensor)
            mask = self.create_mask(src_sentence_tensor)
            input_tok = torch.tensor([trg_sos_idx] * src_sentence_tensor.size(0), device=device)
            outputs = []
            for t in range(max_len):
                pred, hidden, cell, attn = self.decoder(input_tok, hidden, cell, encoder_outputs, mask=mask)
                top1 = pred.argmax(1)
                outputs.append(top1.cpu().numpy())
                input_tok = top1
            # outputs: list of arrays length max_len
            return np.stack(outputs, axis=1)  # [batch, max_len]


ENC_EMB = 256
DEC_EMB = 256
HID_DIM = 512
enc = Encoder(input_dim=src_vocab_size, emb_dim=256, hid_dim=512)
dec = Decoder(output_dim=tgt_vocab_size, emb_dim=256, enc_hid_dim=512*2, dec_hid_dim=512)
model = AttnSeq2Seq(enc, dec, pad_idx=0).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# Prepare data for PyTorch
def to_tensor(arr):
	return torch.tensor(arr, dtype=torch.long)

trainX_tensor = to_tensor(trainX)
trainY_tensor = to_tensor(trainY)
testX_tensor = to_tensor(testX)
testY_tensor = to_tensor(testY)


# Training loop
epochs = 30
batch_size = 64
train_losses = []
val_losses = []
for epoch in range(epochs):
	model.train()
	epoch_loss = 0
	for i in range(0, len(trainX_tensor), batch_size):
		src_batch = trainX_tensor[i:i+batch_size].to(device)
		tgt_batch = trainY_tensor[i:i+batch_size].to(device)
		optimizer.zero_grad()
		output = model(src_batch, tgt_batch)
		output = output.view(-1, tgt_vocab_size)
		tgt_batch = tgt_batch.view(-1)
		loss = criterion(output, tgt_batch)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
	avg_loss = epoch_loss / (len(trainX_tensor) // batch_size + 1)
	train_losses.append(avg_loss)

	# Validation
	model.eval()
	with torch.no_grad():
		val_loss = 0
		for i in range(0, len(testX_tensor), batch_size):
			src_batch = testX_tensor[i:i+batch_size].to(device)
			tgt_batch = testY_tensor[i:i+batch_size].to(device)
			output = model(src_batch, tgt_batch)
			output = output.view(-1, tgt_vocab_size)
			tgt_batch = tgt_batch.view(-1)
			loss = criterion(output, tgt_batch)
			val_loss += loss.item()
		avg_val_loss = val_loss / (len(testX_tensor) // batch_size + 1)
		val_losses.append(avg_val_loss)

	print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

# Save model and vocabs for inference
torch.save(model.state_dict(), 'savepoints/model_seq2seq.pt')
with open('savepoints/src_vocab.pkl', 'wb') as f:
	pickle.dump(src_vocab, f)
with open('savepoints/tgt_vocab.pkl', 'wb') as f:
	pickle.dump(tgt_vocab, f)
with open('savepoints/lengths.pkl', 'wb') as f:
	pickle.dump({'src_length': src_length, 'tgt_length': tgt_length}, f)

# Plot loss
plt.plot(train_losses)
plt.plot(val_losses)
plt.legend(['train', 'validation'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()