import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# open text file and read in data as `text`
with open('shakespeare.txt', 'r') as f:
    text = f.read()


    # Encoding the characters as integers makes it easier to use as input in the network.
    #thus the text is converted as follows:
    # we create two dictionaries:
    # 1. int2char, which maps integers to characters
    # 2. char2int, which maps characters to unique integers
    chars = tuple(set(text))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}

    # encode the text
    encoded = np.array([char2int[ch] for ch in text])

    #LSTM expects an input that is one-hot encoded meaning that each character is converted
    #into an integer (via our created dictionary) and then converted into a column vector where only it's
    #corresponding integer index will have the value of 1 and the rest of the vector will be filled with 0's.

    import torch
    print(torch.__version__)
    def one_hot_encode(arr, n_labels):

        # Initialize the the encoded array
        one_hot = np.zeros((np.multiply(*arr.shape), n_labels), dtype=np.float32)

        # Fill the appropriate elements with ones
        one_hot[np.arange(one_hot.shape[0]), arr.flatten()] = 1.

        # Finally reshape it to get back to the original array
        one_hot = one_hot.reshape((*arr.shape, n_labels))

        return one_hot

    # check that the function works as expected
    test_seq = np.array([[3, 5, 1]])
    one_hot = one_hot_encode(test_seq, 8)

    print(one_hot)


    #to train the data I'll create minibatches >>>
    #I'll take the encoded characters (passed in as the arr parameter) and split them into multiple sequences,
    #given by batch_size.
    # Each of our sequences will be seq_length long
    #batch contains N*M characters, where N is the batch size (the number of sequences in a batch)
    #and M is the seq_length
    #to get the total number of batches, $K$, that we can make from the array arr,
    #you divide the length of arr by the number of characters per batch. Once
    #then split arr into N batches..

    def get_batches(arr, batch_size, seq_length):
        '''Create a generator that returns batches of size
           batch_size x seq_length from arr.

           Arguments
           ---------
           arr: Array you want to make batches from
           batch_size: Batch size, the number of sequences per batch
           seq_length: Number of encoded chars in a sequence
        '''

        batch_size_total = batch_size * seq_length
        # total number of batches we can make
        n_batches = len(arr)//batch_size_total

        # Keep only enough characters to make full batches
        arr = arr[:n_batches * batch_size_total]
        # Reshape into batch_size rows
        arr = arr.reshape((batch_size, -1))

        # iterate through the array, one sequence at a time
        for n in range(0, arr.shape[1], seq_length):
            # The features
            x = arr[:, n:n+seq_length]
            # The targets, shifted by one
            y = np.zeros_like(x)
            try:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n+seq_length]
            except IndexError:
                y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
            yield x, y


# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else:
    print('No GPU available, training on CPU; consider making n_epochs very small.')



############################################
############NETWORK#########################
############################################

#Model Structure
#Create and store the necessary dictionaries
#Define an LSTM layer that takes as params:
#nput size (the number of characters), a hidden layer size n_hidden,
#number of layers n_layers, a dropout probability drop_prob, and a batch_first boolean
#(True, since we are batching)
#Define a dropout layer with dropout_prob
#Define a fully-connected layer with params
#Finally, initialize the weights
class CharRNN(nn.Module):

    def __init__(self, tokens, n_hidden=256, n_layers=2, drop_prob=0.5, lr=0.001):
        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr

        # creating character dictionaries
        self.chars = tokens
        self.int2char = dict(enumerate(self.chars))
        self.char2int = {ch: ii for ii, ch in self.int2char.items()}

        ##define the LSTM
        self.lstm = nn.LSTM(len(self.chars), n_hidden, n_layers,dropout=drop_prob, batch_first=True)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(n_hidden, len(self.chars))


    def forward(self, x, hidden):
        ''' Forward pass through the network.
            These inputs are x, and the hidden/cell state `hidden`. '''

        ## Get the outputs and the new hidden state from the lstm
        r_output, hidden = self.lstm(x, hidden)

        ## TODO: pass through a dropout layer
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)

        ## TODO: put x through the fully-connected layer
        out = self.fc(out)

        # return the final output and the hidden state
        return out, hidden


    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                  weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden


################################################
##########################TRAIN FUNCTION########
################################################
def train(net, data, epochs,  seq_length, lr, batch_size=10, clip=5, val_frac=0.1, print_every=1):
    ''' Training a network

        Arguments
        ---------

        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss

    '''
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data)*(1-val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if(train_on_gpu):
        net.cuda()

    counter = 0
    n_chars = len(net.chars)
    loss_dict={}
    val_dict={}
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        for x, y in get_batches(data, batch_size, seq_length):
            counter += 1

            # One-hot encode our data and make them Torch tensors
            x = one_hot_encode(x, n_chars)
            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if(train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size*seq_length))
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, batch_size, seq_length):
                    # One-hot encode our data and make them Torch tensors
                    x = one_hot_encode(x, n_chars)
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if(train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size*seq_length))

                    val_losses.append(criterion(output, targets.view(batch_size*seq_length)).item())

                net.train() # reset to train mode after iterationg through validation data

                print("Epoch: {}".format(e+1),
                      "Loss: {:.4f}...".format(loss.item()),
			"Validation loss: {}".format(criterion(output, targets.view(batch_size*seq_length)).item()))
                loss_dict[e]= loss.item()
                val_dict[e]=criterion(output, targets.view(batch_size*seq_length)).item()
    return loss_dict, val_dict
#######################################################
# define and print the net
n_hidden=512
n_layers=2

net = CharRNN(chars, n_hidden, n_layers)
print(net)
######################################################
#################TRAINING#############################
######################################################
batch_size = [200,128,64,40]
seq_length = [100,200,250,350]
rates=[0.001,0.005,0.01,0.05]
epchs =  [130,100,60,40]# start small if you are just testing initial behavior

# train the model
for i in range(0,len(rates)):
    print("rate:{}".format(rates[i]), "batch_size:{}".format(batch_size[i]), "Epochs:{}".format(epchs[i]) )
    x,y = train(net, encoded, epochs=epchs[i], batch_size=batch_size[i], seq_length=seq_length[i], lr=rates[i])
    tt= list(x.keys())
    lloss= list(x.values())
    t2= list(y.keys())
    vlloss=list(y.values())
    fig= plt.figure(figsize=(6,8))
    plt.title('Loss vs Validation loss')
    plt.plot(tt, lloss, color='red', marker= "x", label="Text generation loss with learning rate{}".format(rates[i]))
    plt.plot(t2, vlloss, color='green', label="Text generation validation loss with learning rate{}".format(rates[i]))
    plt.legend()
    plt.savefig('Loss vs Validation loss of LR{}.png'.format(rates[i]))
    plt.clf()
