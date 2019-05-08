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
TEACHER_FORCING_RATIO = 0.8
empty_input_ratio = 0.1
KLD_weight = 0.0
LR = 0.05

TENSE = ['normal', 'simplepresent', 'presentprogressive', 'simplepast']


#Encoder
class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, cond_size, cond_embedding_size, latent_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.cond_embedding_size = cond_embedding_size

        self.embedding_word = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden, count=0):
        embeded = self.embedding_word(input).view(1, 1, -1)
        outs, h = self.gru(embeded, hidden)              # (1, 1, hidden_size), (1, 1, hidden_size)

        return h, outs

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size - self.cond_embedding_size, device=device)


#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, cond_size, cond_embedding_size, latent_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding_word = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden, teacher_forcing=False):
        embeded = self.embedding_word(input).view(1, 1, -1)
        embeded = F.relu(embeded)
        # (1, 1, hidden_size), (1, 1, hidden_size)
        outs, h = self.gru(embeded, hidden)
        # (1, vocab_size)
        outputs = self.out(outs).view(-1, self.vocab_size)
        
        return h, outputs


class CVAE(nn.Module):
    def __init__(self, vocab_size, hidden_size, cond_size, cond_embedding_size, latent_size):
        super(CVAE, self).__init__()
        self.latent_size = latent_size
        self.kl_weight = KLD_weight
        
        self.encoder = EncoderRNN(vocab_size, hidden_size, cond_size, cond_embedding_size, latent_size)
        self.decoder = DecoderRNN(vocab_size, hidden_size, cond_size, cond_embedding_size, latent_size)
        self.embedding_cond = nn.Embedding(cond_size, cond_embedding_size)
        self.lat2hid = nn.Linear(latent_size+cond_embedding_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, latent_size)
        self.logvar_linear = nn.Linear(hidden_size, latent_size)

    def KLD(self, mean, log_var):
        # kl divergence loss
        return -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    def forward(self, input_and_cond, target_and_cond, test=False, show=False):
        input_tensor, input_cond = input_and_cond
        target_tensor, target_cond = target_and_cond
        sos_tensor = torch.tensor([SOS_token], dtype=torch.long, device=device)
        eos_tensor = torch.tensor([EOS_token], dtype=torch.long, device=device)
        
        loss = 0
        #----------sequence to sequence part for encoder----------#
        # h = hidden + condition = HIDDEN_SIZE
        encoder_c = self.embedding_cond(input_cond).view(1,1,-1)
        # (1, 1, hidden_size)
        encoder_h = torch.cat((self.encoder.initHidden(), encoder_c), dim=2)
        for i in range(input_tensor.size(0)):
            encoder_h, encoder_outs = self.encoder(input_tensor[i], encoder_h)

        # reparameterization trick
        sample = torch.normal(
            torch.FloatTensor([0]*self.latent_size), 
            torch.FloatTensor([1]*self.latent_size)
        ).to(device)
        mean = self.mean_linear(encoder_outs)
        logvar = self.logvar_linear(encoder_outs)
        encoder_latent = sample.mul(torch.exp(logvar/2)).add_(mean)
        # z = sample * torch.exp(logvar/2) + mean

        gt = torch.cat((target_tensor, eos_tensor))
        use_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False
        #----------sequence to sequence part for decoder----------#
        decoder_c = self.embedding_cond(target_cond).view(1,1,-1)
        # (1,1,latent_size + condition_size)
        lat = torch.cat((encoder_latent, decoder_c), dim=2)
        hidden = self.lat2hid(lat)

        if test:
            pred = None
            decoder_input = sos_tensor
            decoder_hidden = hidden
            for i in range(gt.size(0)):
                decoder_hidden, decoder_output = self.decoder(decoder_input, decoder_hidden, target_cond)
                decoder_input = decoder_output.max(dim=1)[1]
                pred = decoder_input if pred is None else torch.cat((pred, decoder_input))
                if decoder_input.item() == eos_tensor.item():
                    break
            pred = result_trans(pred)
            gt = result_trans(gt)
            if show:
                # print('BLEU-4:', compute_bleu(pred, gt))
                print('{:>20} -> {:20} ans: {}'.format(result_trans(input_tensor)[0], pred[0], gt[0]))
            return compute_bleu(pred, gt)

        pred = None
        loss = 0
        loss_count = 0
        if use_teacher_forcing:
            decoder_input = torch.cat((sos_tensor, target_tensor))
            for i in range(decoder_input.size(0)):
                hidden, decoder_output = self.decoder(decoder_input[i], hidden)
                loss += F.cross_entropy(decoder_output, gt[i].unsqueeze(0))
                loss_count += 1
                pred = decoder_output if pred is None else torch.cat((pred, decoder_output))
            # loss = self.cal_loss(pred, gt, mean, logvar)
            
        else:
            decoder_input = sos_tensor
            for i in range(gt.size(0)):
                hidden, decoder_output = self.decoder(decoder_input, hidden, target_cond)
                decoder_input = decoder_output.max(dim=1)[1].detach()
                loss += F.cross_entropy(decoder_output, gt[i].unsqueeze(0))
                loss_count += 1
                pred = decoder_output if pred is None else torch.cat((pred, decoder_output))
            # loss = self.cal_loss(pred, gt, mean, logvar)
        loss /= loss_count
        kl_loss = self.KLD(mean, logvar)
        loss += kl_loss * self.kl_weight
        return loss, kl_loss


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


def trainIters(cvae, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    print_kl_total = 0
    plot_loss_total = 0  # Reset every plot_every
    plot_kl_total = 0

    cvae_optimizer = optim.Adam(cvae.parameters())
    pairs = get_training_pairs('train.txt')
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    
    folder_name = datetime.now().strftime("%m-%d_%H-%M") + '_iter_' + str(n_iters)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    full_score_count = 0
    best_score = 0.5
    record_dict = {'loss': [], 'score': [], 'kl': []}

    for iter in range(1, n_iters + 1):
        input_and_cond, target_and_cond = training_pairs[iter - 1]

        cvae_optimizer.zero_grad()
        loss, kl_loss = cvae(input_and_cond, target_and_cond)
        loss.backward()
        cvae_optimizer.step()

        print_loss_total += loss.item()
        print_kl_total += kl_loss.item()
        plot_loss_total += loss.item()
        plot_kl_total += kl_loss.item()

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_kl_avg = print_kl_total / print_every
            print_kl_total = 0
            print('='*70)
            print('%s (%d %d%%)' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100))
            print('loss: {:10.4f} kl_loss: {:10.4f}'.format(print_loss_avg, print_kl_avg))
            score = eval(input_and_cond, target_and_cond, cvae, show=True)

            avg_test_score = run_test(cvae)
            save_checkpoint(cvae, folder_name, record_dict)
            if avg_test_score > best_score:
                best_score = avg_test_score
                new_folder = os.path.join(folder_name, 'score_' + str(int(avg_test_score * 100)) + '_iter_' + str(iter))
                if not os.path.exists(new_folder):
                    os.makedirs(new_folder)
                save_checkpoint(cvae, new_folder, record_dict)
        
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_loss_total = 0
            plot_kl_avg = plot_kl_total / plot_every
            plot_kl_total = 0
            record_dict['loss'].append(plot_loss_avg)
            record_dict['kl'].append(plot_kl_avg)
            record_dict['score'].append(run_test(cvae, False))


def save_checkpoint(cvae, folder_name, record_dict):
    torch.save(cvae.state_dict(), os.path.join(folder_name, 'cvae.pkl'))
    file_name = os.path.join(folder_name, 'loss_score.json')
    with open(file_name, 'w') as f:
        json.dump(record_dict, f)


def run_test(cvae, show=True):
    score = 0
    test_pairs = get_training_pairs('test.txt', test=True)
    for test in test_pairs:
        input_and_cond, target_and_cond = tensorsFromPair(test)
        score += eval(input_and_cond, target_and_cond, cvae, show=show)
    if show:
        print('Average BLEU-4:', score / len(test_pairs))
        print('='*70)
    return score / len(test_pairs)
	

def get_training_pairs(input_file, test=False):
    res = []
    if not test:
        all_words = np.loadtxt(input_file, dtype=np.str)
        for four_tense_word in all_words:
            for i in range(4):
                res.append([[four_tense_word[i], i], [four_tense_word[i], i]])
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


def eval(input_and_cond, target_and_cond, cvae, show=False):
    cvae.eval()
    score = cvae(input_and_cond, target_and_cond, test=True, show=show)
    cvae.train()
    return score


def load_checkpoint(cvae, args):
    cvae.load_state_dict(torch.load(os.path.join(args.load, 'cvae.pkl')))
    run_test(cvae)
    
    eos_tensor = torch.tensor([EOS_token], dtype=torch.long, device=device)
    decoder_input = torch.tensor([SOS_token], dtype=torch.long, device=device)
    latent_init = torch.normal(
        torch.FloatTensor([0]*LATENT_SIZE), 
        torch.FloatTensor([1]*LATENT_SIZE)
    ).to(device).view(1,1,-1)
    
    for i in range(len(TENSE)):
        pred = None
        target_cond = torch.tensor([i], dtype=torch.long, device=device)
        decoder_c = cvae.embedding_cond(target_cond).view(1,1,-1)
        lat = torch.cat((latent_init, decoder_c), dim=2)
        decoder_hidden = cvae.lat2hid(lat)
        for i in range(20):
            decoder_hidden, decoder_output = cvae.decoder(decoder_input, decoder_hidden)
            decoder_input = decoder_output.max(dim=1)[1]
            pred = decoder_input if pred is None else torch.cat((pred, decoder_input))
            if decoder_input.item() == eos_tensor.item():
                break
        pred = result_trans(pred)
        print('{}'.format(pred[0]))


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-l", "--load", help="folder path where encoder.pkl & decoder.pkl exists", type=str, default='')
    parser.add_argument("-it", "--iter", help="folder path where encoder.pkl & decoder.pkl exists", type=str, default='')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cvae_init = CVAE(VOCAB_SIZE, HIDDEN_SIZE, len(TENSE), COND_EMB_SIZE, LATENT_SIZE).to(device)
    if args.load:
        load_checkpoint(cvae_init, args)
    else:
        trainIters(cvae_init, 200000, print_every=1000)

    sys.stdout.flush()
    sys.exit()
