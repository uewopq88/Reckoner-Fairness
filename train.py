import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import os
from model import mlp, NoiseGenerator
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
from data_process import load_COMPAS, load_newAdult, load_train_test_valid
from tabulate import tabulate
from identification import identf_split
import torch.nn.functional as F
from collections import OrderedDict
import argparse

def display(f1, bi_acc):
    print("Binary accuracy on test set is %.4f"%(bi_acc))
    print("F1-score on test set is %.4f"%(f1))

def group_comp(df_pred,label,privileged_group):

    g1 = privileged_group
    g0 = privileged_group^1

    #privileged group
    df_priv = df_pred[df_pred[label]==g1]
    priv_truth = df_priv['true_labels']
    priv_pred = df_priv['predicted_labels']

    pr1=len([i for i in priv_pred if i==1])/len(priv_pred)
    cm = confusion_matrix(priv_truth,priv_pred)
    tn1, fp1, fn1, tp1=cm.ravel()
    g1_results = [ f1_score(priv_truth,priv_pred,average='weighted'), tp1/(tp1+fn1), fp1/(fp1+tn1), pr1]
    print("this is pr1: ", pr1)

    # non-privileged group
    df_nopriv = df_pred[df_pred[label]==g0]
    nopriv_truth = df_nopriv['true_labels']
    nopriv_pred = df_nopriv['predicted_labels']

    pr0=len([i for i in nopriv_pred if i==1])/len(nopriv_pred)
    cm = confusion_matrix(nopriv_truth,nopriv_pred)
    tn0, fp0, fn0, tp0 = cm.ravel()
    g0_results = [f1_score(nopriv_truth,nopriv_pred,average='weighted'), tp0/(tp0+fn0), fp0/(fp0+tn0), pr0]
    print("this is pr0: ", pr0)

    # print the summary of comprision
    table = [['Group', 'F1', 'TPR', 'FPR', 'PR'], ['Privileged']+g1_results, ['Non-privileged']+g0_results]
    print(tabulate(table, floatfmt='.3f', headers = "firstrow", tablefmt='psql'))

    #eop=tp0/(tp0+fn0)-tp1/(tp1+fn1)
    eodds= abs((tp0/(tp0+fn0)-tp1/(tp1+fn1))*0.5+(fp0/(fp0+tn0)-fp1/(fp1+tn1))*0.5)
    sp = abs(pr0-pr1)
    #print("Equal Opportunity %.4f"%(eop))
    print("Equal Odds %.4f" %(eodds))
    print("Demographic Parity %.4f"%(sp))

def loss_function(predicted, y):
    clf_loss = F.binary_cross_entropy(predicted, y)

    return clf_loss

def loss_for_low(predicted, input_predicted):
    clf_loss = F.binary_cross_entropy(predicted, input_predicted)

    return clf_loss

def indentify(options, train_set, valid_set, f):
    gpu = options['gpu']
    device = torch.device("cpu")
    input_size = options['input_size']
    hidden_size = options['hidden_size']
    output_size = options['output_size']
    lr = options['lr']
    batch_size = options['batch_size']
    epochs = options['epochs']
    run_id = options['run_id']
    signiture = options['signiture']
    model_path = options['model_path']

    if f == 'highConf' or f == 'lowConf':
        epochs = epochs // 10

    train_size = train_set.__len__()
    valid_size = valid_set.__len__()

    model = mlp(input_size, hidden_size, output_size)

    if gpu:
        device = torch.device("cuda")
        model.to(device)
        DTYPE = torch.FloatTensor
    else:
        DTYPE = torch.FloatTensor

    model_path = os.path.join(
        model_path, "model_{}_{}_{}.pt".format(signiture, f, run_id))
    print("Temp location for models: {}".format(model_path))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    print("Model initialized")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=lr, weight_decay = 0)

    # setup training
    min_valid_loss = float('Inf')
    train_iterator = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_iterator = DataLoader(valid_set, batch_size=batch_size, num_workers=0, shuffle=True)

    print('this is epochs:', epochs)
    for e in range(epochs):
        start_time = time.time()
        model.train()
        avg_train_loss = 0.0
        for batch in train_iterator:
            model.zero_grad()
            x = Variable(batch[0][:, :-1].float().type(DTYPE), requires_grad=False).to(device)
            y = Variable(batch[-1].view(-1, 1).float().type(DTYPE), requires_grad=False).to(device)
            predicted = model(x)
            loss = loss_function(predicted, y)
            avg_loss = loss.item()
            avg_train_loss += avg_loss / train_size
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
        if e % 100 == 0:
            print("Epoch {} complete! Average Training loss: {}".format(e, avg_train_loss))

        model.eval()
        avg_valid_loss = 0.0
        for batch in valid_iterator:
            x = Variable(batch[0][:, :-1].float().type(DTYPE), requires_grad=False).to(device)
            y = Variable(batch[-1].view(-1, 1).float().type(DTYPE), requires_grad=False).to(device)
            predicted = model(x)
            valid_loss = loss_function(predicted, y)
            avg_valid_loss += valid_loss.item()

        predicted = predicted.cpu().data.numpy().reshape(-1, 1)
        predicted = np.round(predicted)
        y = y.cpu().data.numpy().reshape(-1, 1)

        avg_valid_loss = avg_valid_loss / valid_size

        valid_macro_f1 = f1_score(y, predicted, average='macro')
        valid_micro_f1 = f1_score(y, predicted, average='micro')
        if e % 100 == 0:
            print("Validation loss is: {:.4f}".format(avg_valid_loss))
            print("Macro_f1 F1-score on validation set is: {:.4f}".format(valid_macro_f1))
            print("Micro_f1 F1-score on validation set is: {:.4f}".format(valid_micro_f1))

        if (avg_valid_loss < min_valid_loss):
            min_valid_loss = avg_valid_loss
            torch.save(model, model_path)
            print("Found new best model, saving to disk...")
            print("\n")

    print("--- %s seconds ---" % (time.time() - start_time))
    print("\n")

    return model_path

def test(options, model_path, noise_model_path, test_set, f):
    gpu = options['gpu']
    device = torch.device("cpu")
    test_iterator = DataLoader(test_set, batch_size=128, num_workers=0, shuffle=False)

    best_model = torch.load(model_path)
    best_noise_model = torch.load(noise_model_path)
    DTYPE = torch.FloatTensor
    if gpu:
        device = torch.device("cuda")
        best_model = best_model.to(device)
        best_noise_model = best_noise_model.to(device)

    best_model.eval()
    best_noise_model.eval()

    output_test_list = []
    groundtruths = []
    sensitives_list = []
    for batch in test_iterator:
        x = Variable(batch[0][:, :-1].float().type(DTYPE), requires_grad=False).to(device)
        noise = Variable(torch.randn(x.size()).float().type(DTYPE), requires_grad=False).to(device)
        generated_noise = best_noise_model(noise)
        x = x + generated_noise
        y = Variable(batch[-1].view(-1, 1).float().type(DTYPE), requires_grad=False).to(device)
        sensitives = batch[0][:, -1]
        output_test = best_model(x)
        output_test_list += output_test
        groundtruths += y
        sensitives_list += sensitives
    output_test_np = np.round(torch.cat(output_test_list).detach().cpu().numpy())
    groundtruths_np = torch.cat(groundtruths).detach().cpu().numpy()
    results = pd.DataFrame(output_test_np, columns=['predicted_labels'])
    results['true_labels'] = groundtruths_np
    results['sensitive_info'] = sensitives_list
    return output_test_np, groundtruths_np, results

def psudo_learning(options, low_generator_path, psudo_predicted, x_batch):
    gpu = options['gpu']
    device = torch.device("cuda")
    lr = options['lr']
    model_path = options['model_path']
    run_id = options['run_id']
    signiture = options['signiture']
    epochs = 3

    model = torch.load(low_generator_path)

    if gpu:
        device = torch.device("cuda")
        model = model.to(device)
        DTYPE = torch.FloatTensor
    else:
        DTYPE = torch.FloatTensor

    model_path = os.path.join(
        model_path, "model_{}_lowtmp_{}.pt".format(signiture, run_id))
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=lr, weight_decay = 0)

    min_valid_loss = float('Inf')
    for e in range(epochs):
        model.train()
        avg_train_loss = 0.0

        model.zero_grad()
        input_x = x_batch[:-5]
        input_predicted = psudo_predicted[:-5]
        if len(x_batch) < 10:
            input_x = x_batch
            input_predicted = psudo_predicted

        predicted = model(input_x)
        loss = loss_for_low(predicted, input_predicted)

        avg_loss = loss.item()
        avg_train_loss += avg_loss / len(input_x)
        optimizer.zero_grad()

        loss.backward(retain_graph=True)
        optimizer.step()

        if len(x_batch) > 10:
            # for validation
            model.eval()
            avg_valid_loss = 0.0

            predicted = model(x_batch[-5:])
            valid_loss = loss_for_low(predicted, psudo_predicted[-5:])
            avg_valid_loss += valid_loss.item()

            predicted = predicted.cpu().data.numpy().reshape(-1, 1)
            predicted = np.round(predicted)

            avg_valid_loss = avg_valid_loss / len(x_batch[-5:])

            if (avg_valid_loss < min_valid_loss):
                min_valid_loss = avg_valid_loss
                torch.save(model, model_path)

    return model_path

def refinement(options, train_set, valid_set, model, low_generator_path, high_generator_path):
    gpu = options['gpu']
    device = torch.device("cpu")
    lr = options['lr']
    batch_size = options['batch_size']
    epochs = options['epochs']
    run_id = options['run_id']
    signiture = options['signiture']
    model_path = options['model_path']
    input_size = options['input_size']
    hidden_size = options['hidden_size']
    output_size = options['noise_output_size']

    train_size = train_set.__len__()
    valid_size = valid_set.__len__()


    noise_model_path = os.path.join(
        model_path, "model_{}_noise_{}.pt".format(signiture, run_id))
    print("Temp location for models: {}".format(noise_model_path))
    os.makedirs(os.path.dirname(noise_model_path), exist_ok=True)
    noise_model = NoiseGenerator(input_size, hidden_size, output_size)

    if gpu:
        device = torch.device("cuda")
        model = model.to(device)
        noise_model = noise_model.to(device)
        DTYPE = torch.FloatTensor
    else:
        DTYPE = torch.FloatTensor

    print("Model initialized")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=lr, weight_decay = 0)
    parameters_noise = filter(lambda p: p.requires_grad, noise_model.parameters())
    optimizer_noise = optim.Adam(parameters_noise, lr=lr, weight_decay = 0)
    min_valid_loss = float('Inf')
    train_iterator = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_iterator = DataLoader(valid_set, batch_size=batch_size, num_workers=0, shuffle=True)

    print('this is epochs:', epochs)
    for e in range(epochs):
        start_time = time.time()
        model.train()
        noise_model.train()
        avg_train_loss = 0.0
        for batch in train_iterator:
            model.zero_grad()
            noise_model.zero_grad()
            x = Variable(batch[0][:, :-1].float().type(DTYPE), requires_grad=False).to(device)
            noise = Variable(torch.randn(x.size()).float().type(DTYPE), requires_grad=False).to(device)
            generated_noise = noise_model(noise)
            x = x + generated_noise
            y = Variable(batch[-1].view(-1, 1).float().type(DTYPE), requires_grad=False).to(device)

            # first time calulation for psudo training
            predicted = model(x)
            tmp_low_generator_path = psudo_learning(options, low_generator_path, predicted, x)

            #refinement core
            low_generator = torch.load(tmp_low_generator_path)
            lowConf_dict = low_generator.state_dict()
            new_highConf_dict = OrderedDict()
            for key, value in model.state_dict().items():
                if key in lowConf_dict.keys():
                    new_highConf_dict[key] = (
                        lowConf_dict[key] * (1 - 0.999) + value * 0.999
                    )
            model.load_state_dict(new_highConf_dict)

            # get prediction
            predicted = model(x)
            loss = loss_function(predicted, y)

            avg_loss = loss.item()
            avg_train_loss += avg_loss / train_size
            optimizer.zero_grad()
            optimizer_noise.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer_noise.step()

        if e % 100 == 0:
            print("Epoch {} complete! Average Training loss: {}".format(e, avg_train_loss))

        model.eval()
        avg_valid_loss = 0.0
        for batch in valid_iterator:
            x = Variable(batch[0][:, :-1].float().type(DTYPE), requires_grad=False).to(device)
            noise = Variable(torch.randn(x.size()).float().type(DTYPE), requires_grad=False).to(device)
            generated_noise = noise_model(noise)
            x = x + generated_noise
            y = Variable(batch[-1].view(-1, 1).float().type(DTYPE), requires_grad=False).to(device)
            predicted = model(x)
            valid_loss = loss_function(predicted, y)
            avg_valid_loss += valid_loss.item()

        predicted = predicted.cpu().data.numpy().reshape(-1, 1)
        predicted = np.round(predicted)
        y = y.cpu().data.numpy().reshape(-1, 1)

        avg_valid_loss = avg_valid_loss / valid_size

        valid_macro_f1 = f1_score(y, predicted, average='macro')
        valid_micro_f1 = f1_score(y, predicted, average='micro')
        if e % 100 == 0:
            print("Validation loss is: {:.4f}".format(avg_valid_loss))
            print("Macro_f1 F1-score on validation set is: {:.4f}".format(valid_macro_f1))
            print("Micro_f1 F1-score on validation set is: {:.4f}".format(valid_micro_f1))

        if (avg_valid_loss < min_valid_loss):
            min_valid_loss = avg_valid_loss
            torch.save(model, high_generator_path)
            torch.save(noise_model, noise_model_path)
            print("Found new best model, saving to disk...")
            print("\n")

    print("--- %s seconds ---" % (time.time() - start_time))
    print("\n")

    return high_generator_path , noise_model_path


def main(options):
    print(options)
    run_id = options['run_id']
    dataset = options['dataset']
    if dataset == 'COMPAS':
        X, groundtruths = load_COMPAS()
    if dataset == 'NewAdult':
        X, groundtruths = load_newAdult()
    train_set, valid_set, test_set, data_size = load_train_test_valid(X, groundtruths, 'all', 0.50, 0.1)

    print("Training initializing... Setup ID is: {}".format(run_id))
    total_time_start = time.time()

    # Identification stage
    print("Identification stage start")
    hign_id, low_id = identf_split(train_set)
    X_train = train_set.attrs
    y_train = train_set.labels
    high_train = X_train[hign_id]
    high_gt = y_train[hign_id]
    low_train = X_train[low_id]
    low_gt = y_train[low_id]
    h_train_set, h_valid_set, _, _ = load_train_test_valid(high_train, high_gt, 'highConf', 0.9, 0.1)
    l_train_set, l_valid_set, _, _ = load_train_test_valid(low_train, low_gt, 'lowConf', 0.9, 0.1)
    # initialise high conf generator
    high_generator_path = indentify(options, h_train_set, h_valid_set, 'highConf')
    # initialise low conf generator
    low_generator_path = indentify(options, l_train_set, l_valid_set, 'lowConf')
    print("Identification stage completed!")

    # Refinement stage
    print("Refinement stage start")
    high_generator = torch.load(high_generator_path)
    final_model_path, noise_model_path = refinement(options, train_set,
                                                    valid_set, high_generator,
                                                    low_generator_path, high_generator_path)
    print("Refinement stage completed!")

    print('Here is the final results: \n')
    output_test, y, results = test(options, final_model_path, noise_model_path, test_set, 'final')

    bi_acc = accuracy_score(y, np.round(output_test))
    f1 = f1_score(np.round(output_test), y, average='weighted')
    display(f1, bi_acc)
    group_comp(results, 'sensitive_info', 1)
    print("\n")

    print("--- Total time: %s seconds ---" % (time.time() - total_time_start))
    return results, train_set, valid_set, test_set

if __name__ == "__main__":
    options = argparse.ArgumentParser()
    options.add_argument('--run_id', dest='run_id', type=int, default=1)
    options.add_argument('--epochs', dest='epochs', type=int, default=1000)
    options.add_argument('--dataset', dest='dataset', type=str, default='COMPAS') # 'NewAdult'
    options.add_argument('--signiture', dest='signiture', type=str, default='COMPAS_clf')
    options.add_argument('--gpu', dest='gpu', type=bool, default=False)
    options.add_argument('--model_path', dest='model_path',
                         type=str, default='models')
    options.add_argument('--lr', dest='lr', type=float, default=0.001)
    options.add_argument('--batch_size', dest='batch_size', type=int, default=128)
    options.add_argument('--input_size', dest='input_size', type=int, default=32) # [50, 50]
    options.add_argument('--hidden_size', dest='hidden_size', type=int, default=16) #  [32, 32]
    options.add_argument('--output_size', dest='output_size', type=int, default=8) # [16, 50]
    options.add_argument('--noise_output_size', dest='noise_output_size', type=int, default=32) # [16, 50]
    options = vars(options.parse_args())

    results, train_set, valid_set, test_set = main(options)
