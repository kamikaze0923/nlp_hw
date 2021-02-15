from udpos.dataset import create_torch_UDPOS_dataset_and_embedding_layer, EOS_VALUE, PAD_INPUT_WORD_IDX
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from udpos.model import POS_from_WordSeq
from udpos.utils import to_device
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=200,
                    help='Batch size.')
parser.add_argument('--epochs', type=int, default=3,
                    help='Number of training epochs.')
parser.add_argument('--learning-rate', type=float, default=1e-3,
                    help='Learning rate.')
parser.add_argument('--hidden-dim', type=int, default=32,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--lstm-layers', type=int, default=1,
                    help='Number of hidden units in transition MLP.')
parser.add_argument('--embedding-dim', type=int, default=50,
                    help='Dimensionality of embedding.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed (default: 42).')
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.cuda:
    print("Using GPU")
else:
    print("Using CPU")


# Function to enable batch loader to concatenate binary strings of different lengths and pad them
def pad_collate(batch):
    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_rev = [torch.flip(x, dims=(0,)) for x in xx]

    xx_pad = to_device(pad_sequence(xx, batch_first=True, padding_value=PAD_INPUT_WORD_IDX), args)# 0 for 'unk' and 1 for 'pad'
    xx_pad_reverse = to_device(pad_sequence(xx_rev, batch_first=True, padding_value=PAD_INPUT_WORD_IDX), args)# 0 for 'unk' and 1 for 'pad'
    yy_pad = to_device(pad_sequence(yy, batch_first=True, padding_value=EOS_VALUE), args) # 0 for 'SOS' and 1 for 'EOS'

    return xx_pad, xx_pad_reverse, yy_pad, to_device(torch.LongTensor(x_lens), args) # only LongTensor can be used for index selection

def routine_loss(logits, label, criterion=CrossEntropyLoss()):
    """
    :param logits: B x T x TD(tag dimension)
    :param label: B x T
    :param criterion: the pytorch CrossEntropyLoss
    :return: a single loss tensor
    """
    B, T, TD = logits.size()
    return criterion(logits.reshape(-1, TD), label.flatten())


def routine(dataloader, model, optimizer=None):

    avg_loss = 0
    n_exmaple = 0
    for i, batch in enumerate(dataloader):
        print("\r {}/{} batch is training/validating with batch size {} ".format(
            i, dataloader.__len__(), dataloader.batch_size), end="", flush=True
        )
        _, _, yy_pad, _ = batch
        batch_size = yy_pad.size()[0]

        if optimizer:
            optimizer.zero_grad()
            out = model(batch)
            loss = routine_loss(out, yy_pad[:,1:])# match labels except <SOS>
            loss.backward()
            optimizer.step()

        else:
            with torch.no_grad():
                out = model(batch)
                loss = routine_loss(out, yy_pad[:, 1:])# match labels except <SOS>

        avg_loss += loss.item() * batch_size
        n_exmaple += batch_size
    return avg_loss / n_exmaple


def main(args):
    datasets, word_embedding_layer, tag_embedding_layer = create_torch_UDPOS_dataset_and_embedding_layer(args)
    print("Length of datasets train {}, valid {}, test {}".format(len(datasets[0]), len(datasets[1]), len(datasets[2])))
    train_loader = DataLoader(datasets[0], batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate)
    valid_loader = DataLoader(datasets[1], batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)
    test_loader = DataLoader(datasets[2], batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    word_embedding_layer = to_device(word_embedding_layer, args)
    tag_embedding_layer = to_device(tag_embedding_layer, args)
    pos_model = POS_from_WordSeq(args, word_embedding_layer, tag_embedding_layer).to('cuda:0' if args.cuda else 'cpu')
    adam_opt = Adam(params=pos_model.parameters(), lr=args.learning_rate, betas=(0.9, 0.99))

    for name, param in pos_model.named_parameters():
        if param.requires_grad:
            print(name)

    train_loss_buffer = []
    validate_loss_buffer = []

    training_info = f"{args.batch_size}-{args.epochs}-{args.learning_rate}"
    model_info = f"{args.hidden_dim}-{args.lstm_layers}-{args.embedding_dim}"
    hyper_parameters = f"bi_lstm-" + training_info + "-" + model_info

    os.makedirs(os.path.join(args.save_folder, hyper_parameters), exist_ok=True)
    for e in range(args.epochs):
        print("Epoch: {}".format(e))
        train_loss = routine(train_loader, pos_model, optimizer=adam_opt)
        test_loss = routine(valid_loader, pos_model, optimizer=None)
        print("Epoch train loss: {:.3f}, test loss: {:.3f}".format(train_loss, test_loss))
        train_loss_buffer.append(train_loss)
        validate_loss_buffer.append(test_loss)
        torch.save(pos_model.state_dict(), os.path.join(args.save_folder, hyper_parameters, "ckpt_{}.pt".format(e)))
        # print("Unk embedding {}".format(pos_model.encoder.word_embedding_layer.unk_parameter))

    plt.plot(train_loss_buffer)
    plt.plot(validate_loss_buffer)
    plt.xticks([i for i in range(args.epochs)])
    plt.legend(["Train_loss", "Valid_loss"])
    plt.savefig("loss.png")
    plt.close()


if __name__ == "__main__":
    print(vars(args))
    torch.manual_seed(args.seed)
    main(args)

