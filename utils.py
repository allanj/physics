import torch
import torch.autograd as autograd

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))



def read_data(file, number):
    f = open(file, 'r')
    line = f.readline()
    data = []
    while line:

        line = line.rstrip()
        if line != "":
            x = line.split(" ")
            y = f.readline().rstrip().split(" ")
            data.append((x,y))
            if (len(data) == number):
                break
        line = f.readline()
    f.close()
    return data


