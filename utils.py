import torch
import torch.autograd as autograd
import math


def prepare_batch_sequence(seqs, to_ix, batch_size, use_gpu, is_input):
    total = len(seqs)
    num_batches = total // batch_size
    last_batch_size = batch_size
    if total - batch_size * num_batches > 0:
        last_batch_size = total - batch_size * num_batches
        num_batches += 1
    elif total - batch_size * num_batches < 0:
        raise Exception("shouldn't happen")
    # print("num_batches: ", num_batches)
    batches =[]
    for b in range(num_batches):
        batch_idxs = []
        curr_b_size = last_batch_size if b == num_batches - 1 else batch_size
        for i in range(curr_b_size):
            sidx = b * batch_size + i
            seq = seqs[sidx]
            curr_sent = seq[0] if is_input else seq[1]
            idxs = [to_ix[w] for w in curr_sent]
            batch_idxs.append(idxs)
        tensor = torch.LongTensor(batch_idxs)
        if use_gpu:
            tensor = tensor.cuda()
        batches.append(autograd.Variable(tensor))
    return batches

def prepare_cont_batch_sequence(seqs, batch_size, use_gpu, is_input, map_vec = False):
    total = len(seqs)
    num_batches = total // batch_size
    last_batch_size = batch_size
    if total - batch_size * num_batches > 0:
        last_batch_size = total - batch_size * num_batches
        num_batches += 1
    elif total - batch_size * num_batches < 0:
        raise Exception("shouldn't happen")
    # print("num_batches: ", num_batches)
    batches =[]
    for b in range(num_batches):
        batch_vals = []
        curr_b_size = last_batch_size if b == num_batches - 1 else batch_size
        for i in range(curr_b_size):
            sidx = b * batch_size + i
            seq = seqs[sidx]
            curr_sent = seq[0] if is_input else seq[1]
            idxs = [[math.sqrt(1-float(w) * float(w)) ,float(w)] for w in curr_sent] if map_vec and is_input else [[float(w)] for w in curr_sent]
            batch_vals.append(idxs)
        tensor = torch.Tensor(batch_vals)
        # print(tensor.shape)
        if use_gpu:
            tensor = tensor.cuda()
        batches.append(autograd.Variable(tensor))
    return batches




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

def read_all(input_file, output_file, number):
    fi = open(input_file, 'r')
    fo = open(output_file, 'r')
    line_in = fi.readline()
    line_out = fo.readline()
    data = []
    while line_in:
        line_in = line_in.rstrip()
        line_out = line_out.rstrip()
        x = line_in.split(" ")
        y = line_out.split(" ")
        data.append((x,y))
        if (len(data) == number):
            break
        line_in = fi.readline()
        line_out = fo.readline()
    fi.close()
    fo.close()
    return data
