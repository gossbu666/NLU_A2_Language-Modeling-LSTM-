from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import __main__ 

# 1. Define Model Architecture
class LSTMModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.5):
        super(LSTMModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)
        self.nhid = nhid
        self.nlayers = nlayers

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                weight.new_zeros(self.nlayers, bsz, self.nhid))

# 2. Define Vocab Class
class Vocab:
    def __init__(self, token_to_idx, idx_to_token):
        self.stoi = token_to_idx
        self.itos = idx_to_token
        self['<unk>'] 

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get('<unk>'))

    def __len__(self):
        return len(self.stoi)

setattr(__main__, "Vocab", Vocab)

# 3. Load Resources
app = Flask(__name__)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Vocab & Model
try:
    vocab = torch.load('vocab_v2.pth', weights_only = False) # Custom argument to avoid errors
    
    # Load Model
    model = LSTMModel(len(vocab), 200, 200, 2, 0.2)
    model.load_state_dict(torch.load('model_v2.pth', map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    print("✅ Model & Vocab Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model/vocab: {e}")

# Tokenizer
def simple_tokenizer(text):
    return text.lower().split()

# Generation Function
def generate_text_web(prompt, max_words, temperature):
    tokens = simple_tokenizer(prompt)
    indices = [vocab[t] for t in tokens if t in vocab.stoi]
    
    if not indices:
        return f"Error: Prompt '{prompt}' contains unknown words."

    input_seq = torch.tensor(indices, dtype=torch.long).view(-1, 1).to(device)
    hidden = model.init_hidden(1)
    
    generated_words = []
    
    with torch.no_grad():
        output, hidden = model(input_seq, hidden)
        last_logits = output[-1, 0, :]
        
        for _ in range(max_words):
            word_weights = F.softmax(last_logits.div(temperature), dim=0)
            word_idx = torch.multinomial(word_weights, 1).item()
            word = vocab.itos.get(word_idx, '<unk>')
            
            generated_words.append(word)
            if word == '<eos>': break
            
            input_seq = torch.tensor([[word_idx]], dtype=torch.long).to(device)
            output, hidden = model(input_seq, hidden)
            last_logits = output[-1, 0, :]

    return " ".join(generated_words)

# Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    generated_result = ""
    if request.method == 'POST':
        prompt = request.form['prompt']
        try:
            temperature = float(request.form.get('temperature', 1.0))
        except ValueError:
            temperature = 1.0
            
        try:
            generated_result = generate_text_web(prompt, 100, temperature)
            generated_result = f"{prompt} {generated_result}"
        except Exception as e:
            generated_result = f"Error generating text: {str(e)}"
        
    return render_template('index.html', result=generated_result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)