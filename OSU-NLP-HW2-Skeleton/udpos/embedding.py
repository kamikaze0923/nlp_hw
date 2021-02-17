import torch
from udpos.utils import batch_index_select_2d

UNK_IDX = 0
PAD_INPUT_WORD_IDX = 1

def create_glove_corpus_and_embeddings(data_dir="glove_data", dim=50):
    """
    :param data_dir: a directory under "udpos" to store all the glove data files
    :param dim: the dimension of embedding need to be used
    :return:
        embedding matrix: a torch matrix where only the first row is trainable for unknown token in corpus
        embedding mapping: a dictionary maps words in corpus to the index

    # start with the least memory option
    # get called only in dataset.py
    """

    assert dim in [50, 100, 200, 300]
    import os
    import numpy as np
    from pathlib import Path
    current_dir = os.path.dirname(__file__)
    data_dir_abs = os.path.join(current_dir, data_dir)
    os.makedirs(data_dir_abs, exist_ok=True)
    zip_file_name = "glove.6B.zip"
    if zip_file_name not in os.listdir(data_dir_abs):
        print("Downloading Glove data")
        os.system("wget -P {} http://nlp.stanford.edu/data/glove.6B.zip".format(data_dir_abs))
        os.system("unzip {}/{} -d {}".format(data_dir_abs, zip_file_name, data_dir_abs))
    else:
        print("Glove data already downloaded")

    embeddings_dict = {}
    for file in os.listdir(data_dir_abs):
        if str(dim) not in file or not Path(file).suffix == '.txt':
            continue
        with open(os.path.join(data_dir_abs, file), 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                embeddings_dict[word] = vector

    from collections import defaultdict
    embeddings_word2idx = defaultdict(lambda : 0)
    embedding_arrays = []
    for i, (k, v) in enumerate(embeddings_dict.items()):
        embeddings_word2idx[k] = i + len([UNK_IDX, PAD_INPUT_WORD_IDX]) # leave the 0 for unknown words, 1 for pad words
        embedding_arrays.append(v)

    return embeddings_word2idx, torch.FloatTensor(embedding_arrays)

class Glove_Embedding_Layer(torch.nn.Module):

    def __init__(self, embedding_tensor):
        super().__init__()
        self.unk_parameter = torch.nn.Parameter(torch.zeros(size=(1, embedding_tensor.size()[1])), requires_grad=True)
        self.pad_tensor = torch.nn.Parameter(torch.zeros(size=(1, embedding_tensor.size()[1])), requires_grad=False)
        self.embedding_tensor = torch.nn.Parameter(torch.FloatTensor(embedding_tensor), requires_grad=False)
        self.layer_matrix = torch.nn.Parameter(
            torch.cat([self.unk_parameter, self.pad_tensor, self.embedding_tensor]), requires_grad=False
        )# make it parameters, so the to function will move this to GPU
        print("Embedding Layer Created with size {}".format(self.layer_matrix.size()))

    def forward(self, idx):
        # running out of time, will continue to think about how to implement this
        return batch_index_select_2d(self.layer_matrix, idx)


class POS_Embedding_Layer(torch.nn.Module):
    """
    create one-hot embedding for each POS-tag, does not need the gradient
    """

    def __init__(self, tag_dict):
        super().__init__()
        n_tags = len(tag_dict)
        self.layer_matrix = torch.nn.Parameter(torch.FloatTensor(torch.eye(n_tags)), requires_grad=False)
        torch.nn.init.orthogonal_(self.layer_matrix)

    def forward(self, idx):
        return batch_index_select_2d(self.layer_matrix, idx)
