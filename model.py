import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        embed_size, 
        num_layers, 
        heads, 
        device, 
        forward_expansion, 
        dropout, 
        max_length
    ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout,
                    forward_expansion,
                ) for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))
        for layer in self.layers:
            out = layer(out, out, out, mask)
        return out

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(
        self, 
        trg_vocab_size, 
        embed_size, 
        num_layers, 
        heads, 
        forward_expansion, 
        dropout, 
        device, 
        max_length
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device) for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(
        self, 
        src_vocab_size, 
        trg_vocab_size, 
        src_pad_idx, 
        trg_pad_idx, 
        embed_size=256, 
        num_layers=6, 
        forward_expansion=4, 
        heads=8, 
        dropout=0, 
        device="cuda", 
        max_length=100
    ):
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length
        )

        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out

# Sample data: English to French number translations
data = [
    ("one", "un"),
    ("two", "deux"),
    ("three", "trois"),
    ("four", "quatre"),
    ("five", "cinq"),
    ("six", "six"),
    ("seven", "sept"),
    ("eight", "huit"),
    ("nine", "neuf"),
    ("ten", "dix")
]

# Create vocabulary and token mappings
src_vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
trg_vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}

for en, fr in data:
    for token in en.split():
        if token not in src_vocab:
            src_vocab[token] = len(src_vocab)
    for token in fr.split():
        if token not in trg_vocab:
            trg_vocab[token] = len(trg_vocab)

# Reverse vocabulary mappings for decoding
src_inv_vocab = {v: k for k, v in src_vocab.items()}
trg_inv_vocab = {v: k for k, v in trg_vocab.items()}

# Tokenize data
def tokenize(sentence, vocab):
    return [vocab[token] for token in sentence.split()] + [vocab["<EOS>"]]

tokenized_data = [(tokenize(en, src_vocab), tokenize(fr, trg_vocab)) for en, fr in data]

# Pad sequences
def pad_sequence(seq, max_len, pad_idx):
    return seq + [pad_idx] * (max_len - len(seq))

max_len_src = max(len(seq[0]) for seq in tokenized_data)
max_len_trg = max(len(seq[1]) for seq in tokenized_data)

padded_data = [
    (pad_sequence(src, max_len_src, src_vocab["<PAD>"]),
     pad_sequence(trg, max_len_trg, trg_vocab["<PAD>"]))
    for src, trg in tokenized_data
]

# Convert to tensors
src_sentences = torch.tensor([item[0] for item in padded_data])
trg_sentences = torch.tensor([item[1] for item in padded_data])




import torch
import torch.nn as nn
import torch.optim as optim

# Define model parameters
src_vocab_size = len(src_vocab)
trg_vocab_size = len(trg_vocab)
embed_size = 256
num_layers = 3
heads = 8
forward_expansion = 4
dropout = 0.1
max_length = max(max_len_src, max_len_trg)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model
model = Transformer(
    src_vocab_size, trg_vocab_size, 
    src_pad_idx=src_vocab["<PAD>"], trg_pad_idx=trg_vocab["<PAD>"],
    embed_size=embed_size, num_layers=num_layers, forward_expansion=forward_expansion,
    heads=heads, dropout=dropout, device=device, max_length=max_length
).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=src_vocab["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    output = model(src_sentences.to(device), trg_sentences[:, :-1].to(device))
    output = output.reshape(-1, output.shape[2])
    trg = trg_sentences[:, 1:].reshape(-1).to(device)
    
    loss = criterion(output, trg)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    test_sentence = "three"
    tokenized_test_sentence = tokenize(test_sentence, src_vocab)
    padded_test_sentence = pad_sequence(tokenized_test_sentence, max_len_src, src_vocab["<PAD>"])
    src_tensor = torch.tensor(padded_test_sentence).unsqueeze(0).to(device)
    
    trg_indices = [trg_vocab["<SOS>"]]
    for _ in range(max_len_trg):
        trg_tensor = torch.tensor(trg_indices).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, trg_tensor)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indices.append(pred_token)
        if pred_token == trg_vocab["<EOS>"]:
            break
    
    translated_sentence = " ".join(trg_inv_vocab[idx] for idx in trg_indices if idx not in {trg_vocab["<SOS>"], trg_vocab["<EOS>"]})
    print(f'Translated "{test_sentence}" to "{translated_sentence}"')
