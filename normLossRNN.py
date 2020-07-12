import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision
from torch.autograd import Variable
from functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

from dataset import TSNDataSet
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser(description="PyTorch implementation of Temporal Segment Networks")
parser.add_argument('--class_file', type=str, default="classInd.txt")
parser.add_argument('--modality', type=str, default='RGB')

parser.add_argument('--train_source_list', type=str)
parser.add_argument('--train_target_list', type=str)
parser.add_argument('--val_list', type=str)
parser.add_argument('-b', '--batch_size', default=[32, 28, 64], type=int, nargs="+",
                    metavar='N', help='mini-batch size ([source, target, testing])')
parser.add_argument('--copy_list', default=['N', 'Y'], type=str, nargs="+",
                    metavar='N', help='duplicate data in case the dataset is relatively small ([copy source list, copy target list])')
parser.add_argument('--num_segments', type=int, default=5)
parser.add_argument('--val_segments', type=int, default=5)
parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')



global args
args = parser.parse_args()

# EncoderCNN architecture
CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 768
CNN_embed_dim = 512   # latent dim extracted by 2D CNN
dropout_p = 0.0       # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256

# training parameters
k = 101             # number of target category
epochs = 120        # training epochs
batch_size = 40  
learning_rate = 1e-3
log_interval = 10   # interval for displaying training info

norm_beta = 0.01
# Select which frame to begin & end in videos
begin_frame, end_frame, skip_frame = 1, 29, 1

save_model_path = './models/'

def train_source(log_interval, model, device, train_loader, optimizer, epoch):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch

    for batch_idx, (X, y) in enumerate(train_loader):
        # distribute data to device
        X, y = X.to(device), y.to(device).view(-1, )

        N_count += X.size(0)

        optimizer.zero_grad()
        emb = cnn_encoder(X)   # output has dim = (batch, number of classes)
        output = rnn_decoder(emb)
        lossnorm = (emb.norm(p=2, dim=1).mean() - 25) ** 2*norm_beta 
        loss = F.cross_entropy(output, y) + lossnorm
        losses.append(loss.item())

        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tnorm: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),lossnorm.item(), 100 * step_score))

    return losses, scores

def train(log_interval, model, device, source_loader, target_loader, optimizer, epoch):
    
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()

    losses = []
    scores = []
    N_count = 0   # counting total trained sample in one epoch

    for batch_idx, ((X_s, y_s),(X_t, y_t)) in enumerate(zip(source_loader, target_loader)):
        # distribute data to device
        X_s, y_s = X_s.to(device), y_s.to(device).view(-1, )
        X_t, y_t = X_t.to(device), y_t.to(device).view(-1, )

        N_count += X_s.size(0)

        optimizer.zero_grad()
        emb_s = cnn_encoder(X_s)   # output has dim = (batch, number of classes)
        emb_t = cnn_encoder(X_t)   # output has dim = (batch, number of classes)
        output = rnn_decoder(emb)

        lossnorm_s = (emb_s.norm(p=2, dim=1).mean() - 25) ** 2*norm_beta 
        lossnorm_t = (emb_t.norm(p=2, dim=1).mean() - 25) ** 2*norm_beta 
        loss = F.cross_entropy(output, y_s) + lossnorm_s + lossnorm_t
        losses.append(loss.item())


        # to compute accuracy
        y_pred = torch.max(output, 1)[1]  # y_pred != output
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        scores.append(step_score)         # computed on CPU

        loss.backward()
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tnorm_s: {:.6f}\tnorm_t: {:.6f}, Accu: {:.2f}%'.format(
                epoch + 1, N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader), loss.item(),lossnorm_s.item(),lossnorm_t.item(), 100 * step_score))

    return losses, scores


def validation(model, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device).view(-1, )

            output = rnn_decoder(cnn_encoder(X))

            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()                 # sum up batch loss
            y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # collect all y and y_pred in all batches
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # compute accuracy
    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}, Accuracy: {:.2f}%\n'.format(len(all_y), test_loss, 100* test_score))

    # save Pytorch models of best record
    torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    print("Epoch {} model saved!".format(epoch + 1))

    return test_loss, test_score


# Detect devices
use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

#=== Data loading ===#
data_length = 1


# calculate the number of videos to load for training in each list ==> make sure the iteration # of source & target are same
num_source = sum(1 for i in open(args.train_source_list))
num_target = sum(1 for i in open(args.train_target_list))
num_val = sum(1 for i in open(args.val_list))

num_iter_source = num_source / args.batch_size[0]
num_iter_target = num_target / args.batch_size[1]
num_max_iter = max(num_iter_source, num_iter_target)
num_source_train = round(num_max_iter*args.batch_size[0]) if args.copy_list[0] == 'Y' else num_source
num_target_train = round(num_max_iter*args.batch_size[1]) if args.copy_list[1] == 'Y' else num_target

# calculate the weight for each class
class_id_list = [int(line.strip().split(' ')[2]) for line in open(args.train_source_list)]
class_id, class_data_counts = np.unique(np.array(class_id_list), return_counts=True)
class_freq = (class_data_counts / class_data_counts.sum()).tolist()

class_names = [line.strip().split(' ', 1)[1] for line in open(args.class_file)]
num_class = len(class_names)

weight_source_class = torch.ones(num_class)#.cuda()
weight_domain_loss = torch.Tensor([1, 1])#.carameteruda()


# data loading (always need to load the testing data)
val_segments = args.val_segments if args.val_segments > 0 else args.num_segments
val_set = TSNDataSet("", args.val_list, num_dataload=num_val, num_segments=val_segments,
					 new_length=data_length, modality=args.modality,
					 image_tmpl="img_{:05d}.t7",
					 random_shift=False,
					 test_mode=True,
					 )
valid_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size[2], shuffle=False,
										 num_workers=args.workers, pin_memory=True)

source_set = TSNDataSet("", args.train_source_list, num_dataload=num_source_train, num_segments=args.num_segments,
						new_length=data_length, modality=args.modality,
						image_tmpl="img_{:05d}.t7",
						random_shift=False,
						test_mode=True,
						)

source_sampler = torch.utils.data.sampler.RandomSampler(source_set)
source_loader = torch.utils.data.DataLoader(source_set, batch_size=args.batch_size[0], shuffle=False, sampler=source_sampler, num_workers=args.workers, pin_memory=True)

target_set = TSNDataSet("", args.train_target_list, num_dataload=num_target_train, num_segments=args.num_segments,
						new_length=data_length, modality=args.modality,
						image_tmpl="img_{:05d}.t7",
						random_shift=False,
						test_mode=True,
						)

target_sampler = torch.utils.data.sampler.RandomSampler(target_set)

target_loader = torch.utils.data.DataLoader(target_set, batch_size=args.batch_size[1], shuffle=False, sampler=target_sampler, num_workers=args.workers, pin_memory=True)


# Create model
cnn_encoder = Encoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p, CNN_embed_dim=CNN_embed_dim).to(device)
rnn_decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, h_RNN=RNN_hidden_nodes, 
                         h_FC_dim=RNN_FC_dim, drop_p=dropout_p, num_classes=k).to(device)

# Parallelize model to multiple GPUs
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    cnn_encoder = nn.DataParallel(cnn_encoder)
    rnn_decoder = nn.DataParallel(rnn_decoder)

    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.module.fc1.parameters()) + list(cnn_encoder.module.bn1.parameters()) + \
                  list(cnn_encoder.module.fc2.parameters()) + list(cnn_encoder.module.bn2.parameters()) + \
                  list(cnn_encoder.module.fc3.parameters()) + list(rnn_decoder.parameters())

elif torch.cuda.device_count() == 1:
    print("Using", torch.cuda.device_count(), "GPU!")
    # Combine all EncoderCNN + DecoderRNN parameters
    crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
                  list(cnn_encoder.fc2.parameters()) + list(cnn_encoder.bn2.parameters()) + \
                  list(cnn_encoder.fc3.parameters()) + list(rnn_decoder.parameters())

optimizer = torch.optim.Adam(list(cnn_encoder.parameters())+list(rnn_decoder.parameters()), lr=learning_rate)


# record training process
epoch_train_losses = []
epoch_train_scores = []
epoch_test_losses = []
epoch_test_scores = []

# start training
for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train_source(log_interval, [cnn_encoder, rnn_decoder], device, source_loader, optimizer, epoch)

for epoch in range(epochs):
    # train, test model
    train_losses, train_scores = train(log_interval, [cnn_encoder, rnn_decoder], device, source_loader, target_loader, optimizer, epoch)
    
    if epoch%10==0: 
        epoch_test_loss, epoch_test_score = validation([cnn_encoder, rnn_decoder], device, optimizer, valid_loader)

    # save results
    epoch_train_losses.append(train_losses)
    epoch_train_scores.append(train_scores)
    epoch_test_losses.append(epoch_test_loss)
    epoch_test_scores.append(epoch_test_score)

    # save all train test results
    A = np.array(epoch_train_losses)
    B = np.array(epoch_train_scores)
    C = np.array(epoch_test_losses)
    D = np.array(epoch_test_scores)
    np.save('./CRNN_epoch_training_losses.npy', A)
    np.save('./CRNN_epoch_training_scores.npy', B)
    np.save('./CRNN_epoch_test_loss.npy', C)
    np.save('./CRNN_epoch_test_score.npy', D)

