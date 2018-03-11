import torch.nn as nn
import torch.optim as optim
from utils import *
from bilstm import SimpleLSTM

torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"

EMBEDDING_DIM = 2 ## dimension of the embedding
HIDDEN_DIM = 200 ## hidden size
BATCH_SIZE = 1  ## batch size
SENT_LEN = 10  ##Length of sentence, length of vector in this case
NUM_EPOCH = 300 ##number of epochs to use
USE_GPU = False ## if use GPU or not
NUM_TRAIN = 100 ## number of training data to use
NUM_TEST = 100 ##number of test data to use
EVAL_K_EPOCH = 2 ##evaluate the model in every K epoch


def evaluate(model):
    print("[Info] Evaluation on the test set")
    test_tuples = read_data(test_file, NUM_TEST)
    test_inputs = prepare_batch_sequence(test_tuples, word_to_ix, 1, USE_GPU, True)
    test_outputs = prepare_batch_sequence(test_tuples, tag_to_ix, 1, USE_GPU, False)
    corr = 0
    total = 0
    for (test_input, test_output) in zip(test_inputs, test_outputs):
        tag_scores = model(test_input)
        _, max_idxs = tag_scores.max(2)
        for (pred, gold) in zip(max_idxs.data[0], test_output.data[0]):
            if gold == pred:
                corr += 1
            total += 1
    accuracy = corr * 1.0 / total * 100
    print("accuracy: ", corr * 1.0 / total * 100 )
    return accuracy


if __name__ == "__main__":

    train_file = "train.txt"
    test_file = "train.txt"
    training_tuples = read_data(train_file, NUM_TRAIN)

    word_to_ix = {}
    for sentence, tags in training_tuples:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    tag_to_ix = {"0": 0, "1": 1, START_TAG: 2, STOP_TAG: 3}

    training_input = prepare_batch_sequence(training_tuples, word_to_ix, BATCH_SIZE, USE_GPU, True)
    training_output = prepare_batch_sequence(training_tuples, tag_to_ix, BATCH_SIZE, USE_GPU, False)

    if USE_GPU:
        torch.cuda.set_device(0)

    model = SimpleLSTM(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), BATCH_SIZE, SENT_LEN, USE_GPU)
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if USE_GPU:
        model = model.cuda()

    best_accuracy = 0
    best_model = None
    for epoch in range(NUM_EPOCH):
        for batch_input, batch_output in zip(training_input, training_output):
            model.batch_size = len(batch_input)
            model.zero_grad()
            tag_scores = model(batch_input)
            loss = loss_function(tag_scores.view(-1, len(tag_to_ix)), batch_output.view(-1))
            #         print(loss.data[0])
            loss.backward()
            optimizer.step()
        print("Epoch ", epoch, " Loss: ", loss.data[0])
        if (epoch + 1) % EVAL_K_EPOCH == 0:
            curr_accuracy = evaluate(model)
            if curr_accuracy > best_accuracy:
                best_accuracy = curr_accuracy

    evaluate(model)
