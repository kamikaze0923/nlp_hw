import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random
import math
import argparse
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.simplefilter("ignore", UserWarning)

import logging, os

from model import *
from util import calculate_bleu, translate_sentence, save_attention_plot
from routine import train, evaluate

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_logging(file_name=None):
    if file_name:
        try:
            os.remove(file_name)
        except:
            pass

        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        logging.basicConfig(
            filename=file_name,
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

def main(ith_run=0, att="sdp"):
    print(f"The {ith_run}th run ... ")
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn', dest='attn', metavar='a', default=att)
    parser.add_argument('--eval', dest='eval', action='store_true', default=True)
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--beam_size', type=int, default=5)

    args = parser.parse_args()
    if not args.eval:
        set_logging(f"{args.attn}-{ith_run}")
    else:
        set_logging()

    for k,v in vars(args).items():
        print(f"{k}: {v}")

    logging.info('Using device: {}'.format(dev))

    logging.info("Loading tokenizers and dataset")
    spacy_de = spacy.load('de_core_news_sm')#de_core_news_sm.load() 
    spacy_en = spacy.load('en_core_web_sm')#en_core_web_sm.load()

    SRC = Field(tokenize = lambda text: [tok.text for tok in spacy_de.tokenizer(text)], 
                init_token = '<sos>', eos_token = '<eos>', lower = True)

    TRG = Field(tokenize = lambda text: [tok.text for tok in spacy_en.tokenizer(text)], 
                init_token = '<sos>', eos_token = '<eos>', lower = True)

    train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'), fields = (SRC, TRG))

    SRC.build_vocab(train_data, min_freq = 2)
    TRG.build_vocab(train_data, min_freq = 2)

    BATCH_SIZE = 256

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), 
        batch_size = BATCH_SIZE, device = dev,
        sort_key = lambda x : len(x.src))

    # Build model
    logging.info("Building model with attention mechanism: "+args.attn)

    src_vocab_size = len(SRC.vocab)
    dest_vocab_size = len(TRG.vocab)
    word_embed_dim = 128
    enc_hidden_dim = 256
    dec_hidden_dim = 256
    kq_dim = 256
    dropout_rate = 0.5

    if args.attn == "none":
        attn = Dummy(dev=dev)
    elif args.attn == "mean":
        attn = MeanPool()
    elif args.attn == "sdp":
        attn = SingleQueryScaledDotProductAttention(enc_hid_dim=enc_hidden_dim, dec_hid_dim=dec_hidden_dim, kq_dim=kq_dim)

    enc = BidirectionalEncoder(
        src_vocab_size, word_embed_dim, enc_hid_dim=enc_hidden_dim, dec_hid_dim=dec_hidden_dim, dropout=dropout_rate
    )
    dec = Decoder(
        dest_vocab_size, word_embed_dim, enc_hid_dim=enc_hidden_dim, dec_hid_dim=dec_hidden_dim, attention=attn, dropout=dropout_rate
    )
    model = Seq2Seq(enc, dec, dev).to(dev)

    criterion = nn.CrossEntropyLoss(ignore_index = TRG.vocab.stoi[TRG.pad_token])
    if not args.eval:
        print("\n")
        logging.info("Training the model")

        # Set up cross-entropy loss but ignore the pad token when computing it

        optimizer = optim.Adam(model.parameters())

        best_valid_loss = float('inf')

        for epoch in range(args.epoch):

            train_loss = train(model, train_iterator, optimizer, criterion, epoch+1)
            valid_loss = evaluate(model, valid_iterator, criterion, epoch+1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), args.attn+'-best-checkpoint.pt')

            logging.info(f'Epoch: {epoch+1:02}\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
            logging.info(f'Epoch: {epoch+1:02}\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    model.load_state_dict(torch.load(args.attn+'-best-checkpoint.pt', map_location=dev))

    # Test model
    print("\n")
    logging.info("Running test evaluation:")
    test_loss = evaluate(model, test_iterator, criterion, 0)
    bleu = calculate_bleu(test_data, SRC, TRG, model, dev, beam_size=args.beam_size)
    logging.info(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} | Test BLEU {bleu*100:.2f}')

    random.seed(args.seed)
    # torch.manual_seed(args.seed)
    for i in range(10):
        example_id = random.randint(0, len(test_data.examples))
        src = vars(test_data.examples[example_id])['src']
        trg = vars(test_data.examples[example_id])['trg']
        translation, attention = translate_sentence(src, SRC, TRG, model, dev, max_len=50, beam_size=args.beam_size)

        print("\n--------------------")
        print(f'src = {src}')
        print(f'trg = {trg}')
        print(f'prd = {translation}')
        
        save_attention_plot(src, translation, attention, example_id)

    print("\n")
    if args.eval:
        exit(0)
    

if __name__ == "__main__":
    for i in range(3):
        for att in ["sdp", "mean", "none"]:
            main(ith_run=i, att=att)
