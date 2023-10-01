import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

import tokenToolKit
import tokenUtils



def build_mapping_table(filename: str, tokenizer: tokenToolKit.Tokenizer=None):
    
    stoi = dict()   # convert token to index
    itos = dict()   # convert index to token
    
    stoi['.'] = 0
    
    with open(filename, 'r', encoding='utf-8') as file:
        for i, token in enumerate(file.readlines()):
            stoi[token.strip()] = i + 1
    
    if tokenizer is not None:
        for i, token in enumerate(tokenizer.get_tokens()):
            stoi[token.strip()] = i + 1
    
    itos = {i: s for s, i in stoi.items()}
    
    return len(itos), stoi, itos


def build_mapping_table(tokenizer: tokenToolKit.Tokenizer):
    
    stoi = dict()   # convert token to index
    itos = dict()   # convert index to token
    
    stoi['.'] = 0
    
    for i, token in enumerate(tokenizer.get_tokens()):
        stoi[token.strip()] = i + 1
    
    itos = {i: s for s, i in stoi.items()}
    
    return len(itos), stoi, itos



def build_dataset(words, tokenizer: tokenToolKit.Tokenizer=None):   
    
    X, Y = [], []
    
    for word in words:
        context = [0] * context_length
        
        if tokenizer is None:
            for ch in word + '.':
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        else:
            for token in tokenizer.tokenize(word + '.'):
                ix = stoi[token]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]
        
    X = torch.tensor(X)
    Y = torch.tensor(Y)
    
    return X, Y


def train(times=1000, lra: float=0):
    global total_trained_time
    global loss
    
    # batch normalzation
    global bnmean_running, bnstd_running, bngain, bnbias
    
    for i in range(times):
        
        # minibatch construct
        ix = torch.randint(0, Xtr.shape[0], (batch_size, ), generator=g)
        Xb, Yb = Xtr[ix], Ytr[ix]   # batch
    
        # forward pass
        emb = C[Xb]
        embcat = emb.view(emb.shape[0], -1)
        
        # linear layer
        hpreact = embcat @ W1
        
        # batch-norm layer
        bnmeani = hpreact.mean(0, keepdim=True)
        bnstdi = hpreact.std(0, keepdim=True)
        hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias
        
        with torch.no_grad():
            bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
            bnstd_running = 0.999 * bnstd_running + 0.001 * bnstdi
        
        # non-linearity
        h = torch.tanh(hpreact) # hidden layer
        logits = h @ W2 + b2    # output layer
        _loss = F.cross_entropy(logits, Yb)

        # backward pass
        for p in parameters:
            p.grad = None
        _loss.backward()

        # update
        # lr = lrs[i]
        if lra != 0: 
            lr = lra
        else:
            lr = 0.1 if total_trained_time < 1000 \
                else 0.01 if total_trained_time < 10000 else 0.001
                
        l = 0
        for p in parameters:
            l += 1
            p.data += -lr * p.grad

        # track stats
        # lri.append(lre[i])
        stepi.append(i)
    total_trained_time += times
    loss = _loss

# Setters
def set_batch_size(size: int = 32):
    global batch_size
    batch_size = size


# Getters

def get_loss():
    return loss


def get_samples(count=5, filter=lambda _: True):
    
    result = []
    while (count):
        out = []
        context = [0] * context_length
        while True:
            emb = C[torch.tensor([context])]
            
            # linear layer
            hpreact = emb.view(1, -1) @ W1
            # batch-norm layer
            hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
            h = torch.tanh(hpreact)
            
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            if ix == 0: break
        new_word = ''.join(itos[i] for i in out)
        if (filter(new_word)):
            result.append(new_word)
            count -= 1
        else: continue
    return result

# Helper functions
def build_context(prefix: list):
    context = [0] * context_length
    for i in range(len(prefix)):
        context = context[1:] + [stoi[prefix[i]]]
    return context

def get_name_samples(count=5, filter=lambda _: True, prefix=None):
    result = set()
    
    # randomly generate a list of family names
    tokens_list, weights_list = \
        tokenUtils.build_weighted_tables((family_name_frequency, ))    
        
    if prefix is None:
        family_names = \
            random.choices(tokens_list, weights=weights_list, k=count)
    else: family_names = [prefix] * count

    while (count):
        out = []
        # context = [0] * context_length
        
        # Take family name into consideration
        curr_pre = tokenizer.tokenize(family_names[count - 1])
        context = build_context(curr_pre)
        out.extend([stoi[i] for i in curr_pre])
        
        while True:
            emb = C[torch.tensor([context])]
            
            # linear layer
            hpreact = emb.view(1, -1) @ W1
            # batch-norm layer
            hpreact = bngain * (hpreact - bnmean_running) / bnstd_running + bnbias
            h = torch.tanh(hpreact)
            
            logits = h @ W2 + b2
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            context = context[1:] + [ix]
            out.append(ix)
            
            if ix == 0: break
            
        new_name = ''.join(itos[i] for i in out)
        if (filter(new_name)):
            result.add(new_name)
            count -= 1
        else: continue
    return result
        

# Default Filter
length34 = lambda str: len(str) >= 4 and len(str) <= 5
length24 = lambda str: len(str) >= 3 and len(str) <= 5


if __name__ == '__main__':
    # Macros
    random_seed = 42
    generator_seed = 21475982

    word_filename = 'chinese_names_corpus.txt'
    family_name_filename = 'chinese_family_names_corpus.txt'
    family_name_frequency = 'chinese_family_names_frequency.txt'
    character_filename = 'tokens.txt'
    tokenizer_filename = 'tokens.txt'

    tra_per = 0.8   # sample percentage reserved for training
    dev_per = 0.1   # --- for development
    tes_per = 0.1   # --- for testing

    n_hidden = 1000
    n_embd = 10
    context_length = 3
    n_inputs = n_embd * context_length

    batch_size = 32

    # Global Variables
    total_trained_time = 0
    loss = -1
    
    # Tokenizer
    tokenizer = tokenToolKit.TrieTokenizer()
    tokenizer.insertFromFile(tokenizer_filename)

    # Initiate
    names = open(word_filename, 'r', encoding='utf-8').read().splitlines()
    voc_len, stoi, itos = build_mapping_table(tokenizer)
    
    
    # Generate training, development, and test splits
    random.seed(random_seed)
    random.shuffle(names)
    n1 = int(tra_per * len(names))
    n2 = int((tra_per + dev_per) * len(names))


    Xtr, Ytr = build_dataset(names[:n1], tokenizer)
    Xdev, Ydev = build_dataset(names[:n2], tokenizer)
    Xte, Yte = build_dataset(names[n2:], tokenizer)
    

    # Initialize weights and biases
    g = torch.Generator().manual_seed(generator_seed)
    C = torch.randn((voc_len, n_embd), generator=g)
    W1 = torch.randn((n_inputs, n_hidden), generator=g) * (5/3) / (n_inputs ** 0.5)
    # normalization
    # b1 = torch.randn(hidden_layer_neurons, generator=g) * 0.01
    W2 = torch.randn((n_hidden, voc_len), generator=g) * 0.001
    b2 = torch.randn(voc_len, generator=g) * 0

    # Batch normalization
    bngain = torch.ones((1, n_hidden))
    bnbias = torch.zeros((1, n_hidden))
    bnmean_running = torch.zeros((1, n_hidden))
    bnstd_running = torch.ones((1, n_hidden))

    parameters = [C, W1, W2, b2, bngain, bnbias]

    for p in parameters:
        p.requires_grad = True
        
    # Learning rate adjustments
    lri = []
    # lossi = []
    stepi = []