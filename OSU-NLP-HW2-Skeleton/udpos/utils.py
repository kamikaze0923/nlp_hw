import torch

def batch_index_select_2d(matrix_to_select, batch_idx):
    """
    :param matrix_to_select: N x D tensor, N examples to select,  D is the dimensions of the word embedding
    :param batch_idx: B x M tensor, select M examples from N (N >= M)
    :return: B x D tensor
    """

    _, D = matrix_to_select.size()
    B, _ = batch_idx.size()
    matrix_to_select = matrix_to_select.unsqueeze(0).expand(B, -1, -1) # N x D -> B x N x D
    batch_idx = batch_idx.unsqueeze(-1).expand(-1, -1, D) # B x M -> B x M x D
    print(matrix_to_select.device())
    print(batch_idx.device())
    return torch.gather(matrix_to_select, dim=1, index=batch_idx) # B x M x D

def one_hot_vector(categories, hot_index):
    """
    :param categories: an integer of total categories/length of the one_hot vector
    :param hot_index: an integer indicating the one_hot index
    :return: torch.FloatTensor in shape (C,) C is the number of classes
    """
    one_hot = torch.zeros(size=(categories, ), dtype=torch.float32)
    one_hot[hot_index] = 1
    return one_hot
