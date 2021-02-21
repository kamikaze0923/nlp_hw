import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUBLAS_WORKSPACE_CONFIG'] =':16:8' #This is a command to reduce non-deterministic behavior in CUDA
import warnings
warnings.simplefilter("ignore", UserWarning)
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchtext.datasets import LanguageModelingDataset
from torchtext.data import SubwordField, Field, BPTTIterator
from torchtext.data import get_tokenizer
import sys
import argparse
from LanguageModel import LanguageModel
import logging
from torch.distributions import Categorical
from collections import namedtuple
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

torch.no_grad()
  

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--chkpt', dest='chkpt', metavar='c', default="got_language_model")   
  args = parser.parse_args()


  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logging.info('Using device: {}'.format(dev))

  logging.info("Loading tokenizer and vocab from vocab.pkl")
  text_field = pickle.load(open("vocab.pkl", "rb"))
  vocab_size = len(text_field.vocab.itos)

  logging.info("Loading checkpoint {}".format(args.chkpt))
  lm = LanguageModel(vocab_size).to(dev)
  lm.load_state_dict(torch.load(args.chkpt, map_location=torch.device('cpu')))
  lm.eval()

  p = "the night is dark and full of terrors"
  
  # Torch is a bit frustrating at times and some things that ought to be deterministic are not. 
  # This is an attempt to resolve that, but it doesn't work 100% of the time
  torch.set_deterministic(True)
  seed = 42
  mlen = 150

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Vanilla Sampling -----------")
  print(sample(lm, text_field, prompt=p, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 0.0001 -------")
  print(sample(lm, text_field, prompt=p, temp=0.0001, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n------- Temp-Scaled Sampling 100 --------")
  print(sample(lm, text_field, prompt=p, temp=100, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-k Sampling 1 -----------")
  print(sample(lm, text_field, prompt=p, k=1, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-k Sampling 20 -----------")
  print(sample(lm, text_field, prompt=p, k=20, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 0.001 -----------")
  print(sample(lm, text_field, prompt=p, p=0.001, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 0.75 -----------")
  print(sample(lm, text_field, prompt=p, p=0.75, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Top-p Sampling 1 -----------")
  print(sample(lm, text_field, prompt=p, p=1, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=1 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=1, max_len=mlen))

  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=10 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=10, max_len=mlen))
  #
  torch.manual_seed(seed); np.random.seed(seed)
  print("\n----------- Beam Search B=50 -----------")
  print(beamsearch(lm, text_field, prompt=p, beams=50, max_len=mlen))

############################################################################################
# TASK 1.1
############################################################################################
def flatten_list(l):
  if len(l) == 0:
    return []
  else:
    return l[0] + flatten_list(l[1:])

def expand_beams(sws, text_filed, one_beam, one_out, h_t, c_t, n_successor):
  # no need to expand all, just the the beam size is enough. Use flood fill could be even much more vaster here.
  scores = F.log_softmax(one_out, dim=-1)
  sorted_scores = scores.sort(dim=-1, descending=True)

  successors = []
  for idx, score in zip(sorted_scores.indices.squeeze()[:n_successor], sorted_scores.values.detach_().squeeze()[:n_successor]):
    new_sentence = [w for w in one_beam.sentence] + [reverseNumeralize([idx], text_filed)]
    new_score = float(one_beam.score + score)
    new_beam = sws(new_sentence, new_score, h_t, c_t)
    successors.append(new_beam)
  return successors

def beamsearch(model, text_field, beams=5, prompt="", max_len=50):
  sws = namedtuple('Sentence_with_Score', ['sentence', 'score', "h_t", "c_t"])
  out, decodedString, h_0, c_0 = warm_up(model, text_field, prompt)

  all_beams = [sws(decodedString, 0., h_0, c_0)] # all beams has the same score 0, offset by a constant
  all_beams = [expand_beams(sws, text_field, b, out, h_0, c_0, n_successor=beams) for b in all_beams]
  all_beams = flatten_list(all_beams)

  while len(all_beams[0].sentence) < max_len:
    new_all_beams = []
    for one_beam in all_beams:
      last_token = text_field.process([text_field.tokenize(one_beam.sentence[-1].lower())])
      out, h_t, c_t = model(last_token, one_beam.h_t, one_beam.c_t)
      out = out[[-1], :, :]  # continue only with the last output
      new_all_beams.append(expand_beams(sws, text_field, one_beam, out, h_t, c_t, n_successor=beams))
    all_beams = flatten_list(new_all_beams)
    all_beams = sorted(all_beams, reverse=True, key=lambda t: t.score)[:beams]

  return " ".join(all_beams[0].sentence)

############################################################################################
# TASK 1.2
############################################################################################

def sample(model, text_field, prompt="", max_len=50, temp=1.0, k=0, p=1.0):
  assert (k==0 or p==1.0), "Cannot combine top-k and top-p sampling"

  out, decodedString, h_0, c_0 = warm_up(model, text_field, prompt)
  sample_idx = softmax_sample(out, temp, k, p)
  decode = reverseNumeralize(sample_idx, text_field).split()
  decodedString.extend(decode)
  while len(decodedString) < max_len:
    out, h_0, c_0 = model(sample_idx, h_0, c_0)
    out = out[[-1], :, :] # continue only with the last output
    sample_idx = softmax_sample(out, temp, k, p)
    decode = reverseNumeralize(sample_idx, text_field).split()
    decodedString.extend(decode)

  return " ".join(decodedString)

############################################################################################

def warm_up(model, text_field, prompt): # no sample, no beam search
  h_0 = torch.zeros(size=(model.rnn.num_layers, 1, model.rnn.hidden_size)) # batch_size = 1
  c_0 = torch.zeros(size=(model.rnn.num_layers, 1, model.rnn.hidden_size)) # batch_size = 1
  decodedString = []

  if not prompt:
    prompt = np.random.choice(list(text_field.vocab.freqs.keys()))

  decodedString.extend(prompt.split())
  p_tokens = text_field.process([text_field.tokenize(prompt.lower())])
  out, h_0, c_0 = model(p_tokens, h_0, c_0)
  out = out[[-1], :, :]  # continue only with the last output

  return out, decodedString, h_0, c_0


def reverseNumeralize(numeralized_string, text_field):
  strings = [text_field.vocab.itos[i] for i in numeralized_string]
  return " ".join(strings)

def softmax_sample(logits, temp=1.0, k=0, p=1.0): # logits in T x B x HD, B always 1
  probs = F.softmax(logits / temp, dim=-1)
  if k != 0:
    assert type(k) == int
    assert k > 0 and p == 1.0
    topk_prob = torch.topk(probs, k, dim=-1)
    topk_idx, topk_values = topk_prob.indices, topk_prob.values.detach_() # T x B x k, T x B x k
    cleared_probs = torch.zeros(size=logits.size(), dtype=torch.float32)
    cleared_probs[:, :, topk_idx] = topk_values / torch.sum(topk_values)
    probs = cleared_probs
  elif p != 1.0:
    assert type(p) == float
    assert k == 0 and p < 1.0
    sorted_probs = probs.sort(dim=-1, descending=True)
    cum_sorted_probs = torch.cumsum(sorted_probs.values, dim=-1)
    min_set_idx_flag = torch.tensor(cum_sorted_probs > p, dtype=torch.float32)
    min_set_idx = torch.argmin(min_set_idx_flag, dim=-1, keepdim=True) # T x B x 1, B is always 1
    thresh_hold_values = torch.gather(sorted_probs.values, dim=-1, index=min_set_idx).detach_() # T x B x 1, B is always 1
    clear_idx = probs < thresh_hold_values
    probs[clear_idx] = 0
    sum_probs = probs.sum(dim=-1, keepdim=True)
    probs = probs / sum_probs
  distribution = Categorical(probs)
  return distribution.sample() # T x B words together, B always 1, the last dimension becomes the index



if __name__ == "__main__":
  main()