import torch
import tqdm as tq

##########################################################################################
# Train / Eval Functions
##########################################################################################

def train(model, iterator, optimizer, criterion, epoch):

    model.train()

    epoch_loss = 0
    pbar = tq.tqdm(desc="Epoch {}".format(epoch), total=len(iterator), unit="batch")

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)


        loss = criterion(output, trg)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        pbar.update(1)

    pbar.close()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, epoch):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg)

            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
    return epoch_loss / len(iterator)