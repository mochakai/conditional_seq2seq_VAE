from __future__ import unicode_literals, print_function, division
import random
import time
import math
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


"""==============================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. The reparameterization trick
2. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
3. Output your results (BLEU-4 score, words)
4. Plot loss/score
5. Load/save weights

There are some useful tips listed in the lab assignment.
You should check them before starting your lab.
================================================================================"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
#----------Hyper Parameters----------#
HIDDEN_SIZE = 256
LATENT_SIZE = 32
COND_EMB_SIZE = 8
#The number of vocabulary
VOCAB_SIZE = 28
teacher_forcing_ratio = 0.5
empty_input_ratio = 0.1
KLD_weight = 0.0
LR = 0.05

TENSE = ['normal', 'simplepresent', 'presentprogressive', 'simplepast']


#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, cond_size, cond_embedding_size, latent_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.cond_size = cond_size
        self.cond_embedding_size = cond_embedding_size
        self.latent_size = latent_size

        self.embedding_cond = nn.Embedding(cond_size, cond_embedding_size)
        self.embedding_word = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, latent_size)

    def forward(self, input, condition, hidden):
        # h = hidden + condition = HIDDEN_SIZE
        c = self.embedding_cond(condition).view(1,1,-1)
        # (1, 1, hidden_size)
        h = torch.cat((hidden, c), dim=2)
        embeded = self.embedding_word(input).view(-1, 1, self.hidden_size)       # (word_len, 1, hidden_size)
        outs, h = self.gru(embeded, h)              # (word_len, 1, hidden_size), (1, 1, hidden_size)
        
        # reparameterization trick
        sample = torch.normal(
            torch.FloatTensor([0]*self.latent_size), 
            torch.FloatTensor([1]*self.latent_size)
        ).to(device)
        mean = self.linear(h)
        logvar = self.linear(h)
        z = sample * torch.exp(logvar/2) + mean
        
        return z

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size - self.cond_embedding_size, device=device)


#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, cond_size, cond_embedding_size, latent_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding_cond = nn.Embedding(cond_size, cond_embedding_size)
        self.embedding_word = nn.Embedding(vocab_size, hidden_size)
        self.lat2hid = nn.Linear(latent_size+cond_embedding_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, latent, condition, teacher_forcing=False):
        c = self.embedding_cond(condition).view(1,1,-1)
        # (1,1,latent_size + condition_size)
        lat = torch.cat((latent, c), dim=2)
        hidden = self.lat2hid(lat)

        # (word_len, 1, hidden_size)
        embeded = self.embedding_word(input).view(-1, 1, self.hidden_size)
        # (word_len, 1, hidden_size), (1, 1, hidden_size)
        outs, h = self.gru(embeded, hidden)

        # (word_len, vocab_size)
        outputs = self.out(outs).view(-1, self.vocab_size)
        
        return outputs


def train(input_and_cond, target_and_cond, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    input_tensor, input_cond = input_and_cond
    target_tensor, target_cond = target_and_cond
    sos_tensor = torch.tensor([SOS_token], dtype=torch.long, device=device)
    eos_tensor = torch.tensor([EOS_token], dtype=torch.long, device=device)
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0

    #----------sequence to sequence part for encoder----------#
    encoder_latent = encoder(input_tensor, input_cond, encoder.initHidden())
    gt = torch.cat((target_tensor, eos_tensor))

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    #----------sequence to sequence part for decoder----------#
    if use_teacher_forcing:
        decoder_input = torch.cat((sos_tensor, target_tensor))
        decoder_output = decoder(decoder_input, encoder_latent, target_cond)
        loss = criterion(decoder_output, gt)
    else:
        pred = None
        decoder_input = sos_tensor
        for i in range(gt.size(0)):
            decoder_output = decoder(decoder_input, encoder_latent, target_cond)
            decoder_input = decoder_output.max(dim=1)[1]
            pred = decoder_output if pred is None else torch.cat((pred, decoder_output))
        loss = criterion(pred, gt)

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def save_checkpoint(encoder, decoder, folder_name, record_dict):
    torch.save(encoder.state_dict(), os.path.join(folder_name, 'encoder.pkl'))
    torch.save(decoder.state_dict(), os.path.join(folder_name, 'decoder.pkl'))
    file_name = os.path.join(folder_name, 'loss_score.json')
    with open(file_name, 'w') as f:
        json.dump(record_dict, f)


def run_test(encoder, decoder):
    score = 0
    test_pairs = get_training_pairs('test.txt', test=True)
    for test in test_pairs:
        score += evaluate(tensorsFromPair(test), encoder, decoder, test=True)
    print('='*70)
    print('Average BLEU-4:', score / len(test_pairs))


def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every


    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    pairs = get_training_pairs('train.txt')
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    folder_name = datetime.now().strftime("%m-%d_%H-%M") + '_iter_' + str(n_iters)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    full_score_count = 0
    best_score = 5
    record_dict = {'loss': [], 'score': []}

    for iter in range(1, n_iters + 1):
        input_and_cond, target_and_cond = training_pairs[iter - 1]

        loss = train(input_and_cond, target_and_cond, encoder, decoder, 
                     encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        record_dict['loss'].append(loss.item())
        record_dict['score'].append(evaluate(training_pairs[iter - 1], encoder, decoder, show=False))

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('='*70)
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            score = evaluate(tensorsFromPair(random.choice(pairs)), encoder, decoder)
            
            save_checkpoint(encoder, decoder, folder_name, record_dict)
            full_score_count = full_score_count+1 if score >= 1 else 0
            if full_score_count > best_score:
                best_score = full_score_count
                new_folder = os.path.join(folder_name, 'score_' + str(full_score_count) + '_iter_' + str(iter))
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                save_checkpoint(encoder, decoder, new_folder, record_dict)

    run_test(encoder, decoder)
	

def get_training_pairs(input_file, test=False):
    res = []
    if not test:
        all_words = np.loadtxt(input_file, dtype=np.str)
        for four_tense_word in all_words:
            for i in range(4):
                for j in range(4):
                    if i != j:
                        res.append([[four_tense_word[i], i], [four_tense_word[j], j]])
    else:
        all_words = np.loadtxt(input_file, dtype=np.str)
        label = make_test_labels()
        for pairs in all_words:
            tense_pair = next(label)
            res.append([[pairs[0], tense_pair[0]], [pairs[1], tense_pair[1]]])
    return res


def make_test_labels():
    input_pairs = [
        [0, 3],
        [0, 2],
        [0, 1],
        [0, 1],
        [3, 1],
        [0, 2],
        [3, 0],
        [2, 0],
        [2, 3],
        [2, 1],
    ]
    for tense_pair in input_pairs:
        yield tense_pair
    return


def tensorsFromPair(pairs):
    res = []
    for input_target in pairs:
        word, cond = input_target
        res.append([
            torch.tensor([ord(c)-ord('a')+2 for c in word], dtype=torch.long, device=device), 
            torch.tensor(cond, dtype=torch.long, device=device)
        ])
    return res


def result_trans(ans):
    return [''.join([chr(c-2+ord('a')) for c in ans]).replace(chr(EOS_token-2+ord('a')),"")]


#compute BLEU-4 score
def compute_bleu(output, reference):
    output = list(output[0])
    reference = list(reference[0])
    cc = SmoothingFunction()
    return sentence_bleu([reference], output,weights=(0.25, 0.25, 0.25, 0.25),smoothing_function=cc.method1)


def evaluate(pair, encoder, decoder, test=False, show=True):
    input_and_cond, target_and_cond = pair
    with torch.no_grad():
        input_tensor, input_cond = input_and_cond
        target_tensor, target_cond = target_and_cond
        sos_tensor = torch.tensor([SOS_token], dtype=torch.long, device=device)
        eos_tensor = torch.tensor([EOS_token], dtype=torch.long, device=device)

        encoder_latent = encoder(input_tensor, input_cond, encoder.initHidden())
        decoder_input = torch.cat((sos_tensor, target_tensor))

        decoder_output = decoder(decoder_input, encoder_latent, target_cond)
        pred = result_trans(decoder_output.max(dim=1)[1])
        gt = result_trans(target_tensor)

        if show:
            if not test:
                print('BLEU-4:', compute_bleu(pred, gt))
            print('{:>20} -> {:20} ans: {}'.format(result_trans(input_tensor)[0], pred[0], gt[0]))
            # print('pred:', pred)
            # print('true:', gt)

        return compute_bleu(pred, gt)


def load_checkpoint(encoder, decoder, args):
    encoder.load_state_dict(torch.load(os.path.join(args.load, 'encoder.pkl')))
    decoder.load_state_dict(torch.load(os.path.join(args.load, 'decoder.pkl')))
    run_test(encoder, decoder)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-l", "--load", help="folder path where encoder.pkl & decoder.pkl exists", type=str, default='')
    parser.add_argument("-it", "--iter", help="folder path where encoder.pkl & decoder.pkl exists", type=str, default='')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    encoder_init = EncoderRNN(VOCAB_SIZE, HIDDEN_SIZE, len(TENSE), COND_EMB_SIZE, LATENT_SIZE).to(device)
    decoder_init = DecoderRNN(VOCAB_SIZE, HIDDEN_SIZE, len(TENSE), COND_EMB_SIZE, LATENT_SIZE).to(device)
    if args.load:
        load_checkpoint(encoder_init, decoder_init, args)
    else:
        trainIters(encoder_init, decoder_init, 200000, print_every=1000)

    sys.stdout.flush()
    sys.exit()
