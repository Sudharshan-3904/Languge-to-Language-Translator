import torch
import torch.nn as nn
import pickle

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

def encode_sentence(sentence, vocab, length):
    tokens = [vocab.get(word, vocab['<UNK>']) for word in sentence.lower().split()]
    if len(tokens) < length:
        tokens += [vocab['<PAD>']] * (length - len(tokens))
    else:
        tokens = tokens[:length]
    return tokens

def decode_indices(indices, vocab):
    inv_vocab = {idx: word for word, idx in vocab.items()}
    return ' '.join([inv_vocab.get(idx, '') for idx in indices if idx != 0])

# Load vocabs and lengths
with open('savepoints/src_vocab.pkl', 'rb') as f:
    src_vocab = pickle.load(f)
with open('savepoints/tgt_vocab.pkl', 'rb') as f:
    tgt_vocab = pickle.load(f)
with open('savepoints/lengths.pkl', 'rb') as f:
    lengths = pickle.load(f)
src_length = lengths['src_length']
tgt_length = lengths['tgt_length']

src_vocab_size = len(src_vocab)
tgt_vocab_size = len(tgt_vocab)

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(src_vocab_size, tgt_vocab_size, src_length, tgt_length).to(device)
model.load_state_dict(torch.load('savepoints/model_seq2seq.pt', map_location=device))
model.eval()

def predict_translation(input_sentence):
    src_seq = encode_sentence(input_sentence, src_vocab, src_length)
    src_tensor = torch.tensor([src_seq], dtype=torch.long).to(device)
    # Start with <PAD> tokens for decoder input
    tgt_input = torch.tensor([[tgt_vocab['<PAD>']] * tgt_length], dtype=torch.long).to(device)
    with torch.no_grad():
        output = model(src_tensor, tgt_input)
        pred_indices = torch.argmax(output, dim=-1).cpu().numpy()[0]
    return decode_indices(pred_indices, tgt_vocab)

if __name__ == "__main__":
    while True:
        input_sentence = input("Enter a sentence to translate: ")
        translation = predict_translation(input_sentence)
        print("Translation:", translation)
