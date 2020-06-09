import torch
import time


def train(net, train_loader, valid_loader, epochs, criterion, optimizer, writer):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.train()

    for epoch in range(epochs):
        running_outputs, running_labels = [], []
        running_loss = 0
        epoch_time = time.time()
        for i, batch in enumerate(train_loader):
            time_to_read_batch = time.time() - epoch_time
            print("Read time: " + str(time_to_read_batch))
            batch_time = time.time()
            seizures, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            outputs = net(seizures)
            loss = criterion(outputs.view(-1), labels.double())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_outputs += torch.round(torch.sigmoid(outputs))
            running_labels += labels
            print("Train time: " + str(time.time()-batch_time))
            epoch_time = time.time()
        net.eval()

        profile_results(running_outputs, running_labels, epoch, "train", writer, running_loss)

        running_outputs, running_labels = [], []
        for i, batch in enumerate(valid_loader):
            seizures, labels = batch[0].to(device), batch[1].to(device)
            outputs = net(seizures)
            running_outputs += torch.round(torch.sigmoid(outputs))
            running_labels += labels

        profile_results(running_outputs, running_labels, epoch, "test", writer)


def print_last_results(epoch, correctness, sensitivity, specificity, loss=None):
    if loss:
        print("epoch: {:d}, loss {:.4f}, Correct: {:.2f}%, Sensitivty {:.2f}%, Specificity {:.2f}%".format(
        epoch+1, loss, correctness*100, sensitivity*100, specificity*100))
    else:
        print("epoch: {:d}, Correct: {:.2f}%, Sensitivty {:.2f}%, Specificity {:.2f}%".format(
        epoch+1, correctness*100, sensitivity*100, specificity*100))


def profile_results(outputs, targets, epoch, name, writer=None, loss=None):
    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(outputs)):
        output = outputs[i][0]
        target = targets[i]

        if output and target:
            tp = tp+1
        elif not output and not target:
            tn = tn+1
        elif not output and target:
            fn = fn+1
        elif output and not target:
            fp = fp + 1

    correctness, sensitivity, specificity = profile_to_measure(tn, fp, fp, fn)
    it = epoch + 1

    if writer:
        writer.add_scalars("Profile/" + name, {
            "tp": tp,
            "tn": tn,
            "fn": fn,
            "fp": fp
        }, it)
        writer.add_scalars("Measures/" + name, {
            "Sensitivity": sensitivity,
            "Specificity": specificity
        }, it)
        writer.add_scalar("Accuracy/" + name, correctness, it)
        if loss:
            writer.add_scalar("Loss/" + name, loss, it)

    print_last_results(epoch, correctness, sensitivity, specificity, loss=loss)


def profile_to_measure(tn, tp, fp, fn):
    correctness = (tn + tp) / (tp + tn + fp + fn)
    try:
        sensitivity = tp/(tp+fn)
    except:
        sensitivity = 0

    try:
        specificity = tn/(tn+fp)
    except:
        specificity = 0

    return correctness, sensitivity, specificity
