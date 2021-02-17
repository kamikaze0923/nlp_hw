import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sn
import pandas as pd
import imageio
import os
import numpy as np

def visualizeSentenceWithTags(example):
    print("\nToken" + "".join([" "]*(15))+ "POS Tag")
    print("---------------------------------")
    for w, t in zip(example['text'], example['udtags']):
        print(w+"".join([" "]*(20-len(w)))+t)


class Training_Info_Buffer:

    def __init__(self):
        self.train_loss_buffer = []
        self.validate_loss_buffer = []
        self.train_acc_buffer = []
        self.validate_acc_buffer = []


def plot_loss(working_dir, buffer, test_result, best_acc_result, args):
    test_loss, test_acc = test_result

    plt.plot(buffer.train_loss_buffer)
    plt.plot(buffer.validate_loss_buffer)
    plt.scatter(args.epochs - 1, test_loss, marker="^", c='k')
    plt.legend(["Train_loss", "Valid_loss", "Test_loss"])
    plt.savefig(os.path.join(working_dir, "loss.png"))
    plt.close()

    best_acc, best_epoch = best_acc_result
    plt.title(f"Best valid acc: {best_acc:.2f} in epoch {best_epoch}")
    plt.plot(buffer.train_acc_buffer)
    plt.plot(buffer.validate_acc_buffer)
    plt.scatter(args.epochs - 1, test_acc, marker="^", c='k')
    plt.legend(["Train_acc", "Valid_acc", "Test_acc"])
    plt.savefig(os.path.join(working_dir, "accuracy.png"))
    plt.close()

def plot_confusion_matrix(m, label_dict, name):
    index = [tag for tag, _ in sorted(label_dict.items(), key=lambda x: x[1])] # sort according to the value(index in matrix)
    assert m.shape[0] == len(index)
    df_cm = pd.DataFrame(m, index=index, columns=index)
    plt.figure(figsize=(8, 6))
    sn.heatmap(df_cm, annot=True, fmt='d')
    plt.savefig(name)
    plt.close()


def to_gif(working_dir, train_image_file_dir, valid_image_file_dir, name):
    images = []
    train_files = sorted(os.listdir(train_image_file_dir), key=lambda x: int(x.split(".")[0]))
    valid_files = sorted(os.listdir(valid_image_file_dir), key=lambda x: int(x.split(".")[0]))
    for filename1, filename2 in zip(train_files, valid_files):
        arry1 = imageio.imread(os.path.join(train_image_file_dir, filename1))
        arry2 = imageio.imread(os.path.join(train_image_file_dir, filename2))
        images.append(np.concatenate([arry1, arry2], axis=1))
    imageio.mimsave(os.path.join(working_dir, f'{name}.gif'), images)
