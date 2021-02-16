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
    # print("\n")
    # print(matrix_to_select.device, batch_idx.device)
    # print("\n")
    return torch.gather(matrix_to_select, dim=1, index=batch_idx) # B x M x D


def to_device(foo, args):
    """
    :param foo: could be torch.tensor or torch.nn.Module (anything can use .to to move the GPU/CPU)
    :param args: use args.cuda to move the foo
    :return: moved foo
    """
    if args.cuda:
        return foo.to("cuda:0") # coulde be torhc
    else:
        return foo.to("cpu")
