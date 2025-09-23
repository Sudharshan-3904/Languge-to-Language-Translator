import numpy as np
import torch
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


MAX_PAIRS = 9000
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

train_src, test_src, train_tgt, test_tgt = train_test_split(
	src_sentences, tgt_sentences, test_size=0.2, random_state=42)


trainX = encode_sequences(train_src, src_vocab, src_length)
trainY = encode_sequences(train_tgt, tgt_vocab, tgt_length)
testX = encode_sequences(test_src, src_vocab, src_length)
testY = encode_sequences(test_tgt, tgt_vocab, tgt_length)

class Seq2Seq(nn.Module):
	def __init__(self, src_vocab_size, tgt_vocab_size, src_length, tgt_length, embed_size=128, hidden_size=256):
		super(Seq2Seq, self).__init__()
		self.encoder_embed = nn.Embedding(src_vocab_size, embed_size, padding_idx=0)
		self.encoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
		self.decoder_embed = nn.Embedding(tgt_vocab_size, embed_size, padding_idx=0)
		self.decoder_lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
		self.fc = nn.Linear(hidden_size, tgt_vocab_size)
		self.tgt_length = tgt_length

	def forward(self, src, tgt):
		enc_emb = self.encoder_embed(src)
		_, (hidden, cell) = self.encoder_lstm(enc_emb)
		dec_emb = self.decoder_embed(tgt)
		dec_out, _ = self.decoder_lstm(dec_emb, (hidden, cell))
		output = self.fc(dec_out)
		return output

model = Seq2Seq(src_vocab_size, tgt_vocab_size, src_length, tgt_length).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=0)

def to_tensor(arr):
	return torch.tensor(arr, dtype=torch.long)

trainX_tensor = to_tensor(trainX)
trainY_tensor = to_tensor(trainY)
testX_tensor = to_tensor(testX)
testY_tensor = to_tensor(testY)

epochs = 50
batch_size = 128
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