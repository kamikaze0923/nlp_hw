
def visualizeSentenceWithTags(example):
    print("\nToken" + "".join([" "]*(15))+ "POS Tag")
    print("---------------------------------")
    for w, t in zip(example['text'], example['udtags']):
        print(w+"".join([" "]*(20-len(w)))+t)
