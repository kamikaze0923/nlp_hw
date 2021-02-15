from torch.utils.data import Dataset
from udpos.visualize import visualizeSentenceWithTags
from udpos.embedding import create_glove_corpus_and_embeddings, POS_Embedding_Layer, Glove_Embedding_Layer, UNK_IDX, PAD_INPUT_WORD_IDX
import torch

SOS_KEY = "<SOS>"
SOS_VALUE = 0
EOS_KEY = "<EOS>"
EOS_VALUE = 1

def create_torch_UDPOS_dataset_and_embedding_layer(args, do_visualize=False):
    # imported by driver_udpos.py

    from torchtext import data
    from torchtext import datasets

    TEXT = data.Field(lower=True)
    UD_TAGS = data.Field(unk_token=None)
    fields = (("text", TEXT), ("udtags", UD_TAGS))

    all_datasets = datasets.UDPOS.splits(fields)

    if do_visualize:
        visualizeSentenceWithTags(vars(all_datasets[0].examples[0]))

    tag_names = create_pos_dict(all_datasets)
    embeddings_word2idx, embedding_arrays = create_glove_corpus_and_embeddings(dim=args.embedding_dim)

    datasets = [UDPOS_pytoch_dataset(d, embeddings_word2idx, tag_names) for d in all_datasets]
    word_embedding_layer = Glove_Embedding_Layer(embedding_arrays)
    tag_embedding_layer = POS_Embedding_Layer(tag_names)

    return datasets, word_embedding_layer, tag_embedding_layer


def create_pos_dict(datasets, do_histogram=True):

    """
    :param datasets: all udpos datasets(training, validation, testing)
    :return: a dictionary mapping the POS tag to an index
    """

    from collections import Counter
    import os
    histogram_name = "part-of-speech_stats.png"
    counter = Counter()
    for dataset in datasets:
        for example in dataset:
            e = vars(example)
            for w, t in zip(e['text'], e['udtags']):
                counter[t] += 1
    if histogram_name not in os.listdir(os.getcwd()) and do_histogram:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 12))
        plt.bar(counter.keys(), counter.values())
        majority_label_and_count = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0]
        print(
            "Majority label: {}, percentage {:.2f}".format(
                majority_label_and_count[0], majority_label_and_count[1] / sum(counter.values())
            )
        )
        plt.savefig(histogram_name)
        plt.close()

    tag_names = {k:i+len([SOS_KEY, EOS_KEY]) for i,k in enumerate(counter.keys())}
    tag_names[SOS_KEY] = SOS_VALUE
    tag_names[EOS_KEY] = EOS_VALUE

    return tag_names


class UDPOS_pytoch_dataset(Dataset):

    def __init__(self, udpos_data, np_embedding_word2idx, tag_label):
        """
        :param udpos_data: the UDPOS dataset
        :param np_embedding_dict: the dictionary maps words in corpus to its index of the embedding array
        :param tag_label: the dictionary maps speech tags to its index
        """
        self.data = udpos_data[:200]
        self.np_embedding_word2idx = np_embedding_word2idx
        self.tag_label = tag_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        e = vars(self.data[idx])
        x, y = [], []
        y.append(self.tag_label.get(SOS_KEY))
        for w, t in zip(e['text'], e['udtags']):
            x.append(self.np_embedding_word2idx[w])
            y.append(self.tag_label[t])
        y.append(self.tag_label.get(EOS_KEY))
        return torch.LongTensor(x), torch.LongTensor(y)
