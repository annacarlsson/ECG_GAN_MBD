# Import modules
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from torchvision import transforms
from torch.autograd.variable import Variable
sns.set(rc={'figure.figsize':(11, 4)})

import datetime 
from datetime import date
import pathlib
today = date.today()

import random
import json as js
import pickle
import os

# Internal project functions
from data import PD_to_Tensor, SineData

# SETTINGS AND CONSTANTS
# Generator
hidden_nodes_g = 50
layers = 2
tanh_layer = False
bidir = True
    
#Params for the Discriminator
minibatch_layer = 0
minibatch_normal_init_ = True
num_cvs = 1
cv1_out= 10
cv1_k = 3
cv1_s = 1
p1_k = 3
p1_s = 2
cv2_out = 5
cv2_k = 3
cv2_s = 1
p2_k = 3
p2_s = 2

# Training parameters
D_rounds = 3
G_rounds = 1
num_epoch = 50
learning_rate = 0.0002

## LOAD DATA
filename = './sinedata_v2.csv'
compose = transforms.Compose([PD_to_Tensor()])
sine_data = SineData(filename ,transform = compose)

batch_size = 50
data_loader = torch.utils.data.DataLoader(sine_data, batch_size=batch_size, shuffle=True)
num_batches = len(data_loader)

seq_length = sine_data[0].size()[0]

# Define minibatch discrimination
class MinibatchDiscrimination(torch.nn.Module):
    
    def __init__(self,input_features,output_features,minibatch_normal_init, hidden_features=16):
        super(MinibatchDiscrimination,self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.hidden_features = hidden_features
        self.T = torch.nn.Parameter(torch.randn(self.input_features,self.output_features, self.hidden_features))
        
        if minibatch_normal_init == True:
            nn.init.normal(self.T, 0,1)
            
    def forward(self,x):
        M = torch.mm(x,self.T.view(self.input_features,-1))
        M = M.view(-1, self.output_features, self.hidden_features).unsqueeze(0)
        M_t = M.permute(1, 0, 2, 3)
        out = torch.sum(torch.exp(-(torch.abs(M - M_t).sum(3))), dim=0) - 1
        return torch.cat([x, out], 1)
    

# Define discriminator
class Discriminator(torch.nn.Module):
    def __init__(self,seq_length,batch_size,minibatch_normal_init, n_features = 1, num_cv = 1, minibatch = 0, cv1_out= 10, cv1_k = 3, cv1_s = 4, p1_k = 3, p1_s = 3, cv2_out = 10, cv2_k = 3, cv2_s = 3 ,p2_k = 3, p2_s = 3):
        super(Discriminator,self).__init__()
        self.n_features = n_features
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.num_cv = num_cv
        self.minibatch = minibatch
        self.cv1_dims = int((((((seq_length - cv1_k)/cv1_s) + 1)-p1_k)/p1_s)+1)
        self.cv2_dims = int((((((self.cv1_dims - cv2_k)/cv2_s) + 1)-p2_k)/p2_s)+1)
        self.cv1_out = cv1_out
        self.cv2_out = cv2_out
        
        #input should be size (batch_size,num_features,seq_length) for the convolution layer
        self.CV1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels = self.n_features, out_channels = int(cv1_out),kernel_size = int(cv1_k), stride = int(cv1_s))
            ,torch.nn.ReLU()
            ,torch.nn.MaxPool1d(kernel_size = int(p1_k), stride = int(p1_s)))
        
        # 2 convolutional layers
        if self.num_cv > 1:
            self.CV2 = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels = int(cv1_out), out_channels = int(cv2_out) ,kernel_size =int(cv2_k), stride = int(cv2_s))
                ,torch.nn.ReLU()
                ,torch.nn.MaxPool1d(kernel_size = int(p2_k), stride = int(p2_s)))
        
            if self.minibatch > 0:
                self.mb1 = MinibatchDiscrimination(self.cv2_dims*cv2_out,self.minibatch, minibatch_normal_init)
                self.out = torch.nn.Sequential(torch.nn.Linear(int(self.cv2_dims*cv2_out)+self.minibatch,1),torch.nn.Sigmoid()) # to make sure the output is between 0 and 1
            else:
                self.out = torch.nn.Sequential(torch.nn.Linear(int(self.cv2_dims*cv2_out),1),torch.nn.Sigmoid()) # to make sure the output is between 0 and 1 
      

        # 1 convolutional layer
        else:
            if self.minibatch > 0 :    
                self.mb1 = MinibatchDiscrimination(int(self.cv1_dims*cv1_out),self.minibatch, minibatch_normal_init)
                self.out = torch.nn.Sequential(torch.nn.Linear(int(self.cv1_dims*cv1_out)+self.minibatch,1),torch.nn.Dropout(0.2),torch.nn.Sigmoid()) # to make sure the output is between 0 and 1
            else:
                self.out = torch.nn.Sequential(torch.nn.Linear(int(self.cv1_dims*cv1_out),1),torch.nn.Sigmoid())  
           

    def forward(self,x):
        x = self.CV1(x.view(self.batch_size,1,self.seq_length))
        
        #2 Convolutional Layers
        if self.num_cv > 1:   
            x = self.CV2(x)
            x = x.view(self.batch_size,-1)
        
            #2 CNN with minibatch discrimination
            if self.minibatch > 0:
                x = self.mb1(x.squeeze())
                x = self.out(x.squeeze())
             
            #2 CNN and no minibatch discrimination
            else:
                x = self.out(x.squeeze())
        
        # 1 Convolutional Layer
        else: 
            x = x.view(self.batch_size,-1)
       
        #1 convolutional Layer and minibatch discrimination
        if self.minibatch > 0:
            x = self.mb1(x)
            x = self.out(x)
        
        #1 convolutional Layer and no minibatch discrimination
        else:
            x = self.out(x)
        
        return x
  

# Define generator
class Generator(torch.nn.Module):
    
    def __init__(self,seq_length,batch_size,n_features = 1, hidden_dim = 50, num_layers = 2, tanh_output = False, bidirectional = False):
        super(Generator,self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.tanh_output = tanh_output
        self.bidirectional = bidirectional
        
        #Checking if the architecture uses a BiLSTM and setting the output parameters as appropriate.
        if self.bidirectional == True:
            self.num_dirs = 2
        else:
            self.num_dirs = 1
        
        self.layer1 = torch.nn.LSTM(input_size = self.n_features, hidden_size = self.hidden_dim, num_layers = self.num_layers,batch_first = True, bidirectional = self.bidirectional )
        self.out = torch.nn.Linear(self.hidden_dim,1) # to make sure the output is between 0 and 1 - removed ,torch.nn.Sigmoid()
    
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers*self.num_dirs, self.batch_size, self.hidden_dim).zero_(), weight.new(self.num_layers*self.num_dirs, self.batch_size, self.hidden_dim).zero_())
        return hidden
    
    def forward(self,x,hidden):
        x,hidden = self.layer1(x.view(self.batch_size,self.seq_length,1),hidden)
        
        if self.bidirectional == True:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)
        
        x = self.out(x)
        return x.squeeze() #,hidden 

# Define noise generator
def noise(batch_size, features):
    noise_vec = torch.randn(batch_size, features)
    return noise_vec

# Define training loop
current_file_path = str(pathlib.Path(__file__).parent.absolute())
path = current_file_path+"/Run_"+str(today.strftime("%d_%m_%Y"))+"_"+ str(datetime.datetime.now().time()).split('.')[0]
os.mkdir(path)

#Initialising the generator and discriminator
generator_1 = Generator(seq_length,batch_size,hidden_dim =  hidden_nodes_g, tanh_output = tanh_layer, bidirectional = bidir)
discriminator_1 = Discriminator(seq_length, batch_size ,minibatch_normal_init = minibatch_normal_init_, minibatch = minibatch_layer,num_cv = num_cvs, cv1_out = cv1_out,cv1_k = cv1_k, cv1_s = cv1_s, p1_k = p1_k, p1_s = p1_s, cv2_out= cv2_out, cv2_k = cv2_k, cv2_s = cv2_s, p2_k = p2_k, p2_s = p2_s)
#Loss function 
loss_1 = torch.nn.BCELoss()

generator_1.train()
discriminator_1.train()

#Defining optimizer
d_optimizer_1 = torch.optim.Adam(discriminator_1.parameters(),lr = learning_rate)
g_optimizer_1 = torch.optim.Adam(generator_1.parameters(),lr = learning_rate)

G_losses = []
D_losses = []
mmd_list = []
series_list = np.zeros((1,seq_length))

for n in tqdm(range(num_epoch)):
    
    for n_batch, sample_data in enumerate(data_loader):
        
        # Train discriminator
        for d in range(D_rounds):
            #Train Discriminator on Fake Data
            discriminator_1.zero_grad()
            h_g = generator_1.init_hidden()
        
            #Generating the noise and label data
            noise_sample = Variable(noise(len(sample_data),seq_length))
            dis_fake_data = generator_1.forward(noise_sample,h_g).detach()
            y_pred_fake = discriminator_1(dis_fake_data)
            loss_fake = loss_1(y_pred_fake,torch.zeros([len(sample_data),1]))
            loss_fake.backward()
        
            #Train Discriminator on Real Data 
            real_data = Variable(sample_data.float())  
            y_pred_real  = discriminator_1.forward(real_data)
            loss_real = loss_1(y_pred_real,torch.ones([len(sample_data),1]))
            loss_real.backward()
            d_optimizer_1.step() #Updating the weights based on the predictions for both real and fake calculations.
        
        #Train Generator  
        for g in range(G_rounds):
            generator_1.zero_grad()
            h_g = generator_1.init_hidden()
            
            noise_sample = Variable(noise(len(sample_data), seq_length))
            
            gen_fake_data = generator_1.forward(noise_sample,h_g)
            y_pred_gen = discriminator_1(gen_fake_data)
            
            error_gen = loss_1(y_pred_gen,torch.ones([len(sample_data),1]))
            error_gen.backward()
            g_optimizer_1.step()
            
    if (n_batch%100 == 0):
        print("\nERRORS FOR EPOCH: "+str(n)+"/"+str(num_epoch)+", batch_num: "+str(n_batch)+"/"+str(num_batches))
        print("Discriminator error: "+str(loss_fake+loss_real))
        print("Generator error: "+str(error_gen))
        
    if n_batch ==( num_batches - 1):
        G_losses.append(error_gen.item())
        D_losses.append((loss_real+loss_fake).item())
            
        #Saving the parameters of the model to file for each epoch
        torch.save(generator_1.state_dict(), path+'/generator_state_'+str(n)+'.pt')
        torch.save(discriminator_1.state_dict(),path+ '/discriminator_state_'+str(n)+'.pt')

        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            h_g = generator_1.init_hidden()
            fake = generator_1(noise(len(sample_data), seq_length),h_g).detach().cpu()

        series_list = np.append(series_list,fake[0].numpy().reshape((1,seq_length)),axis=0)
    
#Dumping the errors and mmd evaluations for each training epoch.
with open(path+'/generator_losses.txts', 'wb') as fp:
    pickle.dump(G_losses, fp)
with open(path+'/discriminator_losses.txt', 'wb') as fp:
    pickle.dump(D_losses, fp)
    
#Plotting the error graph
plt.plot(G_losses,'-r',label='Generator Error')
plt.plot(D_losses, '-b', label = 'Discriminator Error')
plt.title('GAN Errors in Training')
plt.legend()
plt.savefig(path+'/GAN_errors.png')
plt.close()
  
  
#Plot a figure for each training epoch
i = 0
while i < num_epoch:
    if i%3==0:
        fig, ax = plt.subplots(3,1,constrained_layout=True)
        fig.suptitle("Generated fake data")
    for j in range(0,3):
        ax[j].plot(series_list[i][:])
        ax[j].set_title('Epoch '+str(i))
        i = i+1
     
    plt.savefig(path+'/Training_Epoch_Samples_'+str(i)+'.png')
    plt.close(fig)
    
#Checking the diversity of the samples:
generator_1.eval()
h_g = generator_1.init_hidden()
test_noise_sample = noise(batch_size, seq_length)
gen_data= generator_1.forward(test_noise_sample,h_g).detach()
    
plt.title("Generated Sine Waves")
plt.plot(gen_data[random.randint(0,batch_size-1)].tolist(),'-b')
plt.plot(gen_data[random.randint(0,batch_size-1)].tolist(),'-r')
plt.plot(gen_data[random.randint(0,batch_size-1)].tolist(),'-g')
plt.plot(gen_data[random.randint(0,batch_size-1)].tolist(),'-', color = 'orange')
plt.savefig(path+'/Generated_Data_Sample1.png')
plt.close()