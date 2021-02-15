from collections import Counter
from re import sub, compile
import matplotlib.pyplot as plt
import numpy as np


class UnimplementedFunctionError(Exception):
    pass


class Vocabulary:

    def __init__(self, corpus):

        self.to_words = None
        self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
        self.size = len(self.word2idx)

    def most_common(self, k):
        freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
        return [t for t, f in freq[:k]]

    def text2idx(self, text):
        tokens = self.tokenize(text)
        return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

    def idx2text(self, idxs):
        return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]

    ###########################
    ## TASK 1.1           	 ##
    ###########################
    def tokenize(self, text):
        """

        tokenize takes in a string of text and returns an array of strings splitting the text into discrete tokens.

        :params:
        - text: a string to be tokenize, e.g. "The blue dog jumped, but not high."

        :returns:
        - tokens: a list of strings derived from the text, e.g. ["the", "blue", "dog", "jumped", "but", "not", "high"] for word-level tokenization

        """
        if self.to_words is None:
            raise UnimplementedFunctionError("You have not yet implemented build_vocab.")

        words = self.to_words(text)

        tokens = []
        for w in words:
            if w in self.freq:
                tokens.append(w)
            else:
                tokens.append("UNK")

        return tokens

        # # REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
        # raise UnimplementedFunctionError("You have not yet implemented tokenize.")

    ###########################
    ## TASK 1.2            	 ##
    ###########################
    def build_vocab(self, corpus):
        """

        build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

        :params:
        - corpus: a list string to build a vocabulary over

        :returns:
        - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
        - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
        - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

        """
        self.to_words = lambda sentence: sub(r'[^0-9a-zA-Z_\s]+', '', sentence).lower().split()

        def flatten_list(l):
            list_len = len(l)
            if list_len == 1:
                return l[0]
            else:
                return flatten_list(l[:list_len // 2]) + flatten_list(l[list_len // 2:])

        words_in_lists = list(map(self.to_words, corpus))
        words_in_one_list = flatten_list(words_in_lists)
        counter = Counter(words_in_one_list)

        word2idx, idx2word = {}, {}
        for idx, word in enumerate(counter.keys()):
            word2idx[word] = idx
            idx2word[idx] = word

        print("UNK {} in corpus".format("UNK" in word2idx))

        return word2idx, idx2word, counter

        # # REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
        # raise UnimplementedFunctionError("You have not yet implemented build_vocab.")

    ###########################
    ## TASK 1.3              ##
    ###########################
    def make_vocab_charts(self):
        """

        make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details


        """
        sorted_freqencies = sorted([k for k in self.freq.values()], reverse=True)
        x = [i for i, _ in enumerate(sorted_freqencies)]
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))
        ax1.set_title("Token Frequency Distribution")
        ax1.set_xlabel("Token ID(sorted by frequency)")
        ax1.set_ylabel("Frequency")
        ax1.set_yscale("log")
        ax1.plot(x, sorted_freqencies)
        x_50 = [0, len(sorted_freqencies)]
        y_50 = [50, 50]
        ax1.plot(x_50, y_50, color='r', label="freq=50")
        ax1.legend()

        cum_sorted_fraction = np.cumsum(sorted_freqencies) / sum(sorted_freqencies)
        ax2.set_title("Cumulative Fraction Covered")
        ax2.set_xlabel("Token ID(sorted by frequency)")
        ax2.set_ylabel("Fraction of token Occurences Covered")
        ax2.plot(x, cum_sorted_fraction)
        import bisect
        token_50_idx_l = len(sorted_freqencies) - bisect.bisect_left(sorted_freqencies[::-1], 50)
        token_50_idx_r = len(sorted_freqencies) - bisect.bisect_right(sorted_freqencies[::-1], 50)
        token_50_idx = (token_50_idx_l + token_50_idx_r) // 2
        cum_freq_50 = cum_sorted_fraction[token_50_idx]

        def find_step(cum_sorted_fraction):
            if len(cum_sorted_fraction) == cum_freq_50:
                return 0
            else:
                mid_point = len(cum_sorted_fraction) // 2
                if cum_sorted_fraction[mid_point] == cum_freq_50:
                    return 0
                elif cum_sorted_fraction[mid_point] < cum_freq_50:
                    if cum_sorted_fraction[mid_point + 1] > cum_freq_50:
                        return 0
                    else:
                        return find_step(cum_sorted_fraction[mid_point:]) + mid_point
                else:
                    assert cum_sorted_fraction[mid_point] > cum_freq_50
                    if cum_sorted_fraction[mid_point - 1] < cum_freq_50:
                        return -1
                    else:
                        return find_step(cum_sorted_fraction[:mid_point])

        idx_freq_50 = find_step(cum_sorted_fraction)

        x = [idx_freq_50, idx_freq_50]
        y = [0, 1]
        ax2.plot(x, y, color="red", label="fraction={:.2f}".format(cum_freq_50))
        ax2.legend()

        import os
        fre_vis_name = "frequency.png"
        if fre_vis_name not in os.listdir(os.getcwd()):
            plt.savefig(fre_vis_name)

        # # REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
        # raise UnimplementedFunctionError("You have not yet implemented make_vocab_charts.")

