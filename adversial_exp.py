# model
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch.optim as optim
from skimage.io import imread_collection





def train_adervsial_example(criterion, net, optimizer, pic, epochs):
    x_start=torch.tensor(np.random.normal(100,150,(1024*3,1)))
    x_start = x_start.reshape(1, 3, 32, 32)
    x_start = torch.tensor(x_start).float()
    x_start.requires_grad_(True)

    for epoch in range(epochs):  # loop over the dataset multiple times

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(x_start)


        loss = criterion(outputs,torch.flatten(x_start),torch.flatten(pic))



        loss.backward()
        x_start=torch.flatten(x_start)-((torch.flatten(x_start)-
                                         torch.flatten(pic))*1+torch.flatten(x_start.grad))*0.001


        x_start=torch.tensor(x_start).float()
        x_start = x_start.reshape(1, 3, 32, 32)
        x_start.requires_grad_()

    x_start = torch.tensor(x_start).float().reshape(3,32,32)
    return x_start




def imshow(img, noise_img,save=False, name="",grid=False , noiseImage=False):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    noise_img=noise_img.numpy()/2+0.5
    noise_img=noise_img.reshape(3,32,32)


    if grid:
        img=np.transpose(npimg, (1, 2, 0))
        noise_img=np.transpose(noise_img, (1, 2, 0))
        f, axarr = plt.subplots(1, 3)
        axarr[0].imshow(noise_img)
        axarr[1].imshow(img)
        axarr[2].imshow(img-noise_img)
        axarr[0].set_title("Noise Img")
        axarr[1].set_title("Database cat")
        axarr[2].set_title("Noise")
        plt.show()


    if not save:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    if save:
        plt.imsave(name, np.transpose(npimg, (1, 2, 0)))



class SimpleModel(nn.Module):
    """
    very simple model, to be trained on cpu, for code testing.
    """

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.dropout2(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x

    def save(self, path):
        torch.save({'model_state_dict': self.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])

def my_loss_func(output,x_curr,x_traget):
    y_target=torch.zeros(3)
    y_target[0]=20

    loss_1=torch.dist(x_curr,x_traget,2)
    loss_2=torch.dist(y_target,output,2)
    loss_2=loss_2*loss_2
    loss_1=loss_1*loss_1*1/2
    return loss_1+loss_2

def func():

    test = pd.read_pickle("data/dev.pickle")
  
    net = SimpleModel()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.load("data/0.9022.cpkl")
    cats = imread_collection("all_img/2/*.jpg")
    cats_images=[]

    for image in cats:
        img_t=2*(image/255)-1
        img_t = torch.tensor(img_t).float()
        img_t = img_t.permute((2, 0, 1))
        cats_images.append((torch.tensor(img_t), 2))

    cat=cats_images[0][0]
    plt.imshow(cat.reshape(32,32,3))
    noise_img=train_adervsial_example(my_loss_func, net, optimizer, cat, epochs=6000)
    imshow(cat,noise_img,grid=True)





if __name__ == "__main__":
    func()



