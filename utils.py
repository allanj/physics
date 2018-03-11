import torch
import torch.autograd as autograd


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


