import torch.nn as nn
import torch.optim as optim
from utils import *
from cont_bilstm import ContLSTM
import torch.nn.functional as F

torch.manual_seed(1)
START_TAG = "<START>"
STOP_TAG = "<STOP>"

HIDDEN_DIM = 100        # hidden size
BATCH_SIZE = 1         # batch size
SENT_LEN = 20           # length of sentence, length of vector in this case
NUM_EPOCH = 300         # number of epochs to use
USE_GPU = False         # if use GPU or not
NUM_TRAIN = 1000         # number of training data to use
NUM_TEST = 1000           # number of test data to use
EVAL_K_EPOCH = 2        # evaluate the model in every K epoch
LEARNING_RATE = 0.001   # learning rate of the adam optimizer
GRADIENT_NORM = 3       # gradient clipping norm
TRAIN_INPUT_FILE = "data/continuous/cont_input_L=20_M=50000_m=3_m2=2_g=-0.1_g2=0.5_er=0.0.txt"
TRAIN_OUTPUT_FILE = "data/continuous/cont_output_L=20_M=50000_m=3_m2=2_g=-0.1_g2=0.5_er=0.0.txt"
TEST_INPUT_FILE = "data/continuous/cont_input_L=20_M=50000_m=3_m2=2_g=-0.1_g2=0.5_er=0.0.txt"
TEST_OUTPUT_FILE = "data/continuous/cont_output_L=20_M=50000_m=3_m2=2_g=-0.1_g2=0.5_er=0.0.txt"
MAP_VEC = True


def evaluate(model, loss_function):
    print("[Info] Evaluation on the test set")
    total_loss = 0
    model.batch_size = 1
    for (test_input, test_output) in zip(test_inputs, test_outputs):
        tag_scores = model(test_input)
        loss = loss_function(tag_scores.view(-1), test_output.view(-1))
        total_loss += loss
    print("mse loss: ", total_loss.data[0])
    return total_loss.data[0]


if __name__ == "__main__":

    training_tuples = read_all(TRAIN_INPUT_FILE, TRAIN_OUTPUT_FILE, NUM_TRAIN)
    # training_tuples = read_data("train.txt", NUM_TRAIN)

    training_input = prepare_cont_batch_sequence(training_tuples, BATCH_SIZE, USE_GPU, True, MAP_VEC)
    training_output = prepare_cont_batch_sequence(training_tuples, BATCH_SIZE, USE_GPU, False, MAP_VEC)
    print("[Info] Finish reading data.")
    if USE_GPU:
        torch.cuda.set_device(0)

    model = ContLSTM(2 if MAP_VEC else 1, HIDDEN_DIM, BATCH_SIZE, SENT_LEN, USE_GPU)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    if USE_GPU:
        model = model.cuda()

    test_tuples = read_all(TEST_INPUT_FILE, TEST_OUTPUT_FILE, NUM_TEST)
    # test_tuples = read_data("train.txt", NUM_TEST)
    test_inputs = prepare_cont_batch_sequence(test_tuples, 1, USE_GPU, True, MAP_VEC)
    test_outputs = prepare_cont_batch_sequence(test_tuples, 1, USE_GPU, False, MAP_VEC)

    print("[Info] Start training...")
    best_loss = 0
    best_model = None
    for epoch in range(NUM_EPOCH):
        total_loss = 0
        for batch_input, batch_output in zip(training_input, training_output):
            model.batch_size = len(batch_input)
            model.zero_grad()
            tag_scores = model(batch_input)
            loss = loss_function(tag_scores.view(-1), batch_output.view(-1))
            #         print(loss.data[0])
            total_loss += loss.data[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), GRADIENT_NORM, norm_type=2)
            optimizer.step()
        print("Epoch ", epoch, " Loss: ", total_loss)
        if (epoch + 1) % EVAL_K_EPOCH == 0:
            curr_loss = evaluate(model, loss_function)
            if curr_loss < best_loss:
                best_loss = curr_loss

    evaluate(model, loss_function)
