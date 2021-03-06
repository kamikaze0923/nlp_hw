from torch.nn.utils.rnn import pad_sequence
from udpos.dataset import EOS_VALUE, PAD_INPUT_WORD_IDX
import torch
from udpos.utils import to_device
import numpy as np


# Function to enable batch loader to concatenate binary strings of different lengths and pad them
def pad_collate(batch, args):

    (xx, yy) = zip(*batch)
    x_lens = [len(x) for x in xx]
    xx_rev = [torch.flip(x, dims=(0,)) for x in xx]

    xx_pad = to_device(pad_sequence(xx, batch_first=True, padding_value=PAD_INPUT_WORD_IDX), args)# 0 for 'unk' and 1 for 'pad'
    xx_pad_reverse = to_device(pad_sequence(xx_rev, batch_first=True, padding_value=PAD_INPUT_WORD_IDX), args)# 0 for 'unk' and 1 for 'pad'
    yy_pad = to_device(pad_sequence(yy, batch_first=True, padding_value=EOS_VALUE), args) # 0 for 'SOS' and 1 for 'EOS'

    return xx_pad, xx_pad_reverse, yy_pad, to_device(torch.LongTensor(x_lens), args)

def routine_loss(logits, label, x_lens, criterion=torch.nn.CrossEntropyLoss(reduction='sum')):
    """
    :param logits: B x T x TD(tag dimension)
    :param label: B x T
    :param x_lens: B (word/speech tag length)
    :param criterion: the pytorch CrossEntropyLoss
    :return: a single loss tensor
    """

    loss = criterion(logits.permute(0, 2, 1) , label).sum() # label need to be in 2nd dimension to calculate loss
    confusion_matrix = np.zeros(shape=(logits.size()[-1], logits.size()[-1]), dtype=np.int32)
    pred_label = logits.argmax(dim=-1, keepdim=True)
    pred_label = pred_label.squeeze(-1)
    n_words, correct_words = 0, 0
    for i, l in enumerate(x_lens):
        n_words += l + 1
        valid_pred, valid_label = pred_label[i,:l+1], label[i,:l+1]
        # for a, b in zip(valid_pred, valid_label):
        #     # print(a,b)
        #     confusion_matrix[a, b] += 1
        stats = np.zeros(shape=confusion_matrix.shape, dtype=np.int32)
        index = [i for i in zip(valid_pred.detach().cpu().numpy(), valid_label.detach().cpu().numpy())]
        index = [(i*stats.shape[0], i*stats.shape[0]+j) for i,j in index]
        np.put(stats, index, 1) # only work for 1d case
        confusion_matrix = np.add(confusion_matrix, stats)
        flag = torch.eq(valid_label, valid_pred)
        correct_words += torch.sum(flag)
    return loss, correct_words, n_words, confusion_matrix


def routine(dataloader, model, optimizer=None):

    avg_loss = 0
    correct_words = 0
    n_words = 0
    confusion_matrix = None
    for i, batch in enumerate(dataloader):
        print("\r {}/{} batch is training/validating with batch size {} ".format(
            i, dataloader.__len__(), dataloader.batch_size), end="", flush=True
        )
        _, _, yy_pad, x_lens = batch

        if optimizer:
            optimizer.zero_grad()
            out = model(batch)
            loss, batch_correct_word, batch_n_word, confusion_matrix = routine_loss(out, yy_pad[:,1:], x_lens)# match labels except <SOS>
            loss.backward()
            # model.encoder.word_embedding_layer.pad_tensor.grad = None
            # model.encoder.word_embedding_layer.embedding_tensor.grad = None
            optimizer.step()
        else:
            with torch.no_grad():
                out = model(batch)
                loss, batch_correct_word, batch_n_word, confusion_matrix = routine_loss(out, yy_pad[:, 1:], x_lens)# match labels except <SOS>

        avg_loss += loss.item()
        correct_words += batch_correct_word
        n_words += batch_n_word

    return avg_loss / n_words, correct_words / n_words, (correct_words, n_words), confusion_matrix
