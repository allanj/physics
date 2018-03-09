import torch.optim as optim
from utils import *
from bilstmcrf import BiLSTM_CRF


torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 10
HIDDEN_DIM = 200
NUM_TRAIN = 100
NUM_TEST = 100
NUM_EPOCH = 100
EVAL_K_EPCOH = 10
USE_GPU = False
BATCH_SIZE = 100

train_file = "train.txt"
test_file = "test.txt"

def evaluate(model):
    print("[Info] Evaluation on the test set")
    test_data = read_data(test_file, NUM_TEST)
    corr = 0
    total = 0
    for (x,y) in test_data:
        test_sent = prepare_sequence(x, word_to_ix)
        if USE_GPU:
            test_sent = test_sent.cuda()
        result = model(test_sent)[1]
        total += len(y)
        for i in range(len(y)):
            corr += 1 if int(y[i]) == result[i] else 0

    # print("corr: ", corr, " total: ", total)
    accuracy = corr * 1.0 / total * 100
    print("accuracy: ", corr * 1.0 / total * 100 )
    return accuracy


if __name__ == "__main__":
    # Make up some training data
    training_data = read_data(train_file, NUM_TRAIN)
    if USE_GPU:
        torch.cuda.set_device(0)
    word_to_ix = {}
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)

    tag_to_ix = {"0": 0, "1": 1, START_TAG: 2, STOP_TAG: 3}

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, USE_GPU)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
    if USE_GPU:
        model = model.cuda()

    best_accuracy = 0
    best_model = None
    for epoch in range(NUM_EPOCH):
        total_loss = 0
        idx = 0
        for sentence, tags in training_data:
            model.zero_grad()
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.LongTensor([tag_to_ix[t] for t in tags])
            if USE_GPU:
                sentence_in = sentence_in.cuda()
                targets = targets.cuda()
            neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)
            neg_log_likelihood.backward()
            total_loss += neg_log_likelihood.data[0]
            idx += 1
            if idx == BATCH_SIZE:
                idx = 0
                optimizer.step()
        print("Epoch ", epoch, " loss: ", total_loss)
        if (epoch + 1) % EVAL_K_EPCOH == 0:
            curr_accuracy = evaluate(model)
            if curr_accuracy > best_accuracy:
                best_accuracy = curr_accuracy
                # model.save_state_dict('lstmcrf.pt')
    print(best_accuracy)

    evaluate(model)
