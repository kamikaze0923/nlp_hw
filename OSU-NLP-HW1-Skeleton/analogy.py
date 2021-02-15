import gensim.downloader
w2v = gensim.downloader.load('word2vec-google-news-300')


def analogy(a, b, c):
    print(a+" : "+b+" :: "+c+" : ?")
    print([(w, round(c, 3)) for w, c in w2v.most_similar(positive=[c, b], negative=[a])])


if __name__ == "__main__":
    # analogy('man', 'king', 'woman')
    # analogy('iphone', 'kobe', 'android')
    # analogy('starcraft', 'father', 'warcraft')
    # analogy('algebra', 'matrix', 'calculus')
    # analogy('Asia', 'Himalayas', 'America')
    # analogy('Pacific', 'Japan', 'Atlantic')
    analogy('man', 'scientist', 'woman')
    analogy('woman', 'scientist', 'man')
