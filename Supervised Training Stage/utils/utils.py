import numpy as np
import torch
import math


def mat2sixD(mat):
    num = mat.shape[0]
    result = mat[:, :, :2].reshape(num, -1)
    return result


def norm(mat):
    sum = torch.sqrt(torch.sum(torch.square(mat), dim=1))
    sum = sum.unsqueeze(1)
    sum = torch.cat((sum,sum,sum),dim=1)
    result = torch.true_divide(mat, sum)
    return result


def dot(mat1, mat2):
    result = [torch.dot(mat1[i, :], mat2[i, :]) for i in range(mat1.shape[0])]
    result = torch.tensor(result).unsqueeze(1)
    result = torch.cat((result,result,result),dim=1)
    return result


def sixD2mat(sixD):
    sixD = torch.tensor(sixD)
    assert sixD.shape[1] == 6
    num = sixD.shape[0]
    sixD = sixD.reshape(num, 3, 2)
    a1 = sixD[:, :, 0]
    a2 = sixD[:, :, 1]
    b1 = norm(a1)
    b2 = a2 - torch.mul(dot(b1, a2), b1)
    b2 = norm(b2)
    b3 = torch.cross(b1, b2)
    result = torch.cat((b1.unsqueeze(2), b2.unsqueeze(2), b3.unsqueeze(2)), dim=2)
    return result

def quat_diff(quat1, quat2):
    diff = torch.mul(quat1[:, 0], quat2[:, 0]) + torch.mul(quat1[:, 1], quat2[:, 1]) + \
           torch.mul(quat1[:, 2], quat2[:, 2]) + torch.mul(quat1[:, 3], quat2[:, 3])
    diff = torch.acos(diff) / math.pi * 180 * 2
    return torch.mean(diff)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print('Error increases for the' + str(self.counter) + 'times.')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)
        print('Model is saved.')
        self.val_loss_min = val_loss


def write_board(writer, messages, content, epoch):
    assert len(messages) == len(content)
    for i in range(len(messages)):
        writer.add_scalar(messages[i], content[i], epoch + 1)


def accuracy(output, target):
    _, pred = output.topk(1)
    pred = pred.view(-1)

    correct = pred.eq(target).sum()

    return correct.item(), target.size(0) - correct.item()



def print_info(args):
    print('The model has', args.nhead, 'heads;')
    print('the sequence length is', args.seq_length, ';')
    print('the model dimension is', args.d_model, ';')
    print('the encoder has', args.num_encoder_layers, 'layers;')
    print('the decoder has', args.num_decoder_layers, 'layers;')
    print('the batch size is', args.batch_size, ';')
    print('the feedforward layer dimension is', args.dim_feedforward, ';')
    print('the learning rate is', args.lr, '.')
