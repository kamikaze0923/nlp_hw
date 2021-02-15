
from datasets import load_dataset
from Vocabulary import Vocabulary
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.utils.extmath import randomized_svd
import logging
import itertools
from sklearn.manifold import TSNE
import time
import os
import scipy.sparse


import random
random.seed(42)
np.random.seed(42)

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

class UnimplementedFunctionError(Exception):
	pass

def save_to_sparse(name, np_arr):
	print("save {}".format(name))
	a = time.time()
	scp_sparse_matrix = scipy.sparse.csc_matrix(np_arr)
	scipy.sparse.save_npz(name, scp_sparse_matrix)
	b = time.time()
	print("save time {:.2f}".format(b - a))

def load_from_sparse(name):
	print("load {}".format(name))
	a = time.time()
	scp_sparse_matrix = scipy.sparse.load_npz(name)
	arr = scp_sparse_matrix.todense()
	b = time.time()
	print("load time {:.2f}".format(b - a))
	return arr


###########################
## TASK 2.2              ##
###########################

def compute_cooccurrence_matrix(corpus, vocab):
	"""
	    
	    compute_cooccurrence_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns 
	    an N x N count matrix as described in the handout. It is up to the student to define the context of a word

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns: 
	    - C: a N x N matrix where the i,j'th entry is the co-occurrence frequency from the corpus between token i and j in the vocabulary

	"""

	cooccurance_matrix_name = "cooccurrence_matrix.npz"
	COUNT_CENTER_WORD_AS_CONTEXT = True # context word include center word or not
	if cooccurance_matrix_name not in os.listdir(os.getcwd()):

		words_in_lists = list(map(vocab.to_words, corpus))
		from collections import defaultdict
		accumulate_indicies = defaultdict(lambda : 0)
		a = time.time()
		for i,words in enumerate(words_in_lists):
			print("\r{}/{}".format(i, len(words_in_lists)), end="", flush=True)
			for j,_ in enumerate(words):
				context_words_indicies = [vocab.word2idx[w] for k, w in enumerate(words) if k != j or COUNT_CENTER_WORD_AS_CONTEXT]
				center_world_idx = vocab.word2idx[words[j]]
				for r, c in [(center_world_idx, context_words_index) for context_words_index in context_words_indicies]:
					accumulate_indicies[r*vocab.size + c] += 1 # numpy put only works when index is given in 1d number

		indicies_to_put = []
		values_to_put = []
		for k,v in accumulate_indicies.items():
			indicies_to_put.append(k)
			values_to_put.append(v)

		max_occurrence = max(values_to_put)
		print("\nmax number in coocurrence matrix: {}".format(max_occurrence))

		if max_occurrence < 2 ** 8:
			cooccurrence_matrix = np.zeros(shape=(vocab.size, vocab.size), dtype=np.uint8)
		elif max_occurrence < 2 ** 16:
			cooccurrence_matrix = np.zeros(shape=(vocab.size, vocab.size), dtype=np.uint16)
		else:
			cooccurrence_matrix = np.zeros(shape=(vocab.size, vocab.size), dtype=np.uint32)

		print("putting values into cooccurence matrix with size {} ... ".format(cooccurrence_matrix.shape))
		np.put(cooccurrence_matrix, indicies_to_put, values_to_put)
		print("max number in coocurrence matrix after put: {}".format(np.max(cooccurrence_matrix)))

		cooccurrence_matrix = np.asmatrix(cooccurrence_matrix)
		b = time.time()
		print("\ncompute time {:.2f}".format(b-a))

		non_zero_set = set()
		print("checking if matrix is diagonal ... ")
		a = time.time()
		for x, y in zip(*np.where(cooccurrence_matrix != 0)):
			if x == y:
				continue
			if (y, x) in non_zero_set:
				non_zero_set.remove((y, x))
			else:
				non_zero_set.add((x, y))

		assert len(non_zero_set) == 0
		b = time.time()
		print("diagonal matrix check finished, time used {:.2f}".format(b - a))
		save_to_sparse(cooccurrence_matrix, cooccurance_matrix_name)
	else:
		cooccurrence_matrix = load_from_sparse(cooccurance_matrix_name)

	return cooccurrence_matrix

	# # REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
	# raise UnimplementedFunctionError("You have not yet implemented compute_count_matrix.")

###########################
## TASK 2.3              ##
###########################

def compute_ppmi_matrix(corpus, vocab):
	"""
	    
	    compute_ppmi_matrix takes in list of strings corresponding to a text corpus and a vocabulary of size N and returns 
	    an N x N positive pointwise mutual information matrix as described in the handout. Use the compute_cooccurrence_matrix function. 

	    :params:
	    - corpus: a list strings corresponding to a text corpus
	    - vocab: a Vocabulary object derived from the corpus with N words

	    :returns: 
	    - PPMI: a N x N matrix where the i,j'th entry is the estimated PPMI from the corpus between token i and j in the vocabulary

	"""
	ppmi_arr_name = "ppmi_matrix.npz"
	if ppmi_arr_name not in os.listdir(os.getcwd()):
		PPMI = compute_cooccurrence_matrix(corpus, vocab).astype(np.float32)
		print(PPMI.shape, PPMI.dtype, type(PPMI))
		diag = np.asmatrix(np.expand_dims(np.diagonal(PPMI), axis=-1)).astype(np.float32)
		a = time.time()
		print("Computing PPMI ...")
		def work_flow(x, diag, N, EPSL=1e-6):
			print("0, {} {} {}".format(x.shape, x.dtype, type(x)))
			x = np.multiply(x, N)
			print("1, {} {} {}".format(x.shape, x.dtype, type(x)))
			x = np.divide(x, diag)
			print("2, {} {} {}".format(x.shape, x.dtype, type(x)))
			x = np.divide(x, diag.transpose())
			print("3, {} {} {}".format(x.shape, x.dtype, type(x)))
			x = x + EPSL
			print("4, {} {} {}".format(x.shape, x.dtype, type(x)))
			x = np.log(x)
			print("5, {} {} {}".format(x.shape, x.dtype, type(x)))
			x = np.maximum(0, x)
			print("6, {} {} {}".format(x.shape, x.dtype, type(x)))
			return x
		PPMI = work_flow(PPMI, diag, float(len(corpus)))
		b = time.time()
		print(PPMI.shape, PPMI.dtype, type(PPMI))
		print("PPMI compute time {:.2f}".format(b - a))
		save_to_sparse(ppmi_arr_name, PPMI)
	else:
		PPMI = load_from_sparse(ppmi_arr_name)
	return PPMI
	# # REMOVE THIS ONCE YOU IMPLEMENT THIS FUNCTION
	# raise UnimplementedFunctionError("You have not yet implemented compute_ppmi_matrix.")

################################################################################################
# Main Skeleton Code Driver
################################################################################################
def main_freq():
	logging.info("Loading dataset")
	dataset = load_dataset("ag_news")
	dataset_text =  [r['text'] for r in dataset['train']]
	# dataset_labels = [r['label'] for r in dataset['train']]

	logging.info("Building vocabulary")
	vocab = Vocabulary(dataset_text)
	vocab.tokenize("The quick, brown fox jumped over the lazy dog.")
	vocab.make_vocab_charts()
	plt.close()
	plt.pause(0.01)

	logging.info("Computing PPMI matrix")
	PPMI = compute_ppmi_matrix(dataset_text, vocab)

	logging.info("Performing Truncated SVD to reduce dimensionality")
	word_vectors = dim_reduce(PPMI)

	logging.info("Preparing T-SNE plot")
	plot_word_vectors_tsne(word_vectors, vocab, file_name="freq_tsne_matrix.npz")


def dim_reduce(PPMI, k=16):
	word_vector_name = "word_vector.npz"
	if word_vector_name not in os.listdir(os.getcwd()):
		U, Sigma, VT = randomized_svd(PPMI, n_components=k, n_iter=10, random_state=42)
		SqrtSigma = np.sqrt(Sigma)[np.newaxis,:]

		U = U*SqrtSigma
		V = VT.T*SqrtSigma

		word_vectors = np.concatenate( (U, V), axis=1)
		word_vectors = word_vectors / np.linalg.norm(word_vectors, axis=1)[:,np.newaxis]
		print(word_vectors.shape, word_vectors.dtype, type(word_vectors))
		save_to_sparse(word_vector_name, word_vectors)
	else:
		word_vectors = load_from_sparse(word_vector_name)

	return word_vectors


def plot_word_vectors_tsne(word_vectors, vocab, file_name):
	if file_name not in os.listdir(os.getcwd()):
		coords = TSNE(metric="cosine", perplexity=50, random_state=42).fit_transform(word_vectors)
		print(coords.shape, coords.dtype, type(coords))
		save_to_sparse(file_name, coords)
	else:
		coords = load_from_sparse(file_name)

	# plt.cla()
	top_word_idx = vocab.text2idx(" ".join(vocab.most_common(1000)))
	plt.figure(figsize=(256,192))
	plt.plot(coords[top_word_idx,0], coords[top_word_idx,1], 'o', markerfacecolor='none', markeredgecolor='k', alpha=1, markersize=3)
	plt.tick_params(labelsize=288)

	for i in tqdm(top_word_idx):
		plt.annotate(vocab.idx2text([i])[0],
			xy=(coords[i,0],coords[i,1]),
			xytext=(5, 2),
			textcoords='offset points',
			ha='right',
			va='bottom',
			fontsize=15)

	plt.savefig("tsne-{}.png".format(file_name.split("_")[0]))



if __name__ == "__main__":
    main_freq()

