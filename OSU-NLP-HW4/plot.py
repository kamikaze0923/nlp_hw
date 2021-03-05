import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

train_loss_np = np.zeros(shape=(3, 3, 100))
valid_loss_np = np.zeros(shape=(3, 3, 100))

att_type = ['sdp', 'mean', 'none']
colors = ['r', 'b', 'g']


def find_between(s, start, end):
    return (s.split(start))[1].split(end)[0]


plt.figure(figsize=(10, 10))
for i, t in enumerate(att_type):
    for n in range(3):
        file = f"{t}-{n}"
        with open(file, 'r') as f:
            ls = f.readlines()
            train_losses = list(map(
                lambda l: float(find_between(l, 'Loss:', '|')), filter(lambda l: "Epoch" in l and "Train" in l, ls)
            ))
            valid_losses = list(map(
                lambda l: float(find_between(l, 'Loss:', '|')), filter(lambda l: "Epoch" in l and "Val" in l, ls)
            ))
        assert len(train_losses) == 100
        assert len(valid_losses) == 100
        train_loss_np[i, n, :] = train_losses
        valid_loss_np[i, n, :] = valid_losses

    x = np.arange(0, 100)
    y = np.mean(train_loss_np[i], axis=0)
    err = np.std(train_loss_np[i], axis=0)
    plt.errorbar(x=x, y=y, yerr=err, linestyle='solid', label=f"{t}-train", color=colors[i])
    y = np.mean(valid_loss_np[i], axis=0)
    err = np.std(valid_loss_np[i], axis=0)
    plt.errorbar(x=x, y=y, yerr=err, linestyle='dashed', label=f"{t}-validation", color=colors[i])

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")
plt.close()