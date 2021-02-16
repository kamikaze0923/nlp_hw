import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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


def plot_loss(buffer, test_result, args):
    test_loss, test_acc = test_result

    plt.plot(buffer.train_loss_buffer)
    plt.plot(buffer.validate_loss_buffer)
    plt.scatter(args.epochs - 1, test_loss, marker="^", c='k')
    plt.xticks([i for i in range(args.epochs)])
    plt.legend(["Train_loss", "Valid_loss", "Test_loss"])
    plt.savefig("loss.png")
    plt.close()

    plt.plot(buffer.train_acc_buffer)
    plt.plot(buffer.validate_acc_buffer)
    plt.scatter(args.epochs - 1, test_acc, marker="^", c='k')
    plt.xticks([i for i in range(args.epochs)])
    plt.legend(["Train_acc", "Valid_acc", "Test_acc"])
    plt.savefig("accuracy.png")
    plt.close()
