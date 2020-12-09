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


def accuracy_and_confsuion(test_loader, net, test, data):
    indexs = []
    from sklearn.metrics import confusion_matrix
    correct = 0
    total = 0
    predicted_all = []
    labels_all = []
    outputs_prob_vectors = []
    net.eval()
    mis_label = []
    with torch.no_grad():
        for i, dd in enumerate(test_loader):
            images, labels = dd
            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)
            predicted_all.extend(list(predicted.numpy()))
            labels_all.extend(list(labels.numpy()))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # if predicted.item() == 1 and labels.item() == 2:
            #     plt.xlabel("True label :" + str(labels) + "model output " + str(outputs))
            #     plt.ylabel(i)
            #     indexs.append(i)
            #     imshow(test[i][0])
            #     mis_label.append(i)
    conf_matrix = confusion_matrix(labels_all, predicted_all)
    conf_matrix = conf_matrix / conf_matrix.sum(axis=1).reshape(1, 3)
    print(conf_matrix)
    real_acc = (conf_matrix[1, 1] + conf_matrix[2, 2] + conf_matrix[0, 0]) / 3
    print("Real accuracy is : " + str(real_acc))
    if real_acc > 0.88:
        print("saved")
        net.save(str(real_acc)[:6] + ".cpkl")
    print(confusion_matrix(labels_all, predicted_all))

    return conf_matrix


def show_worse_mistakes(criterion, net, test, trainloader):
    all_loss = np.zeros(700)
    trainloader = torch.utils.data.DataLoader(test, batch_size=1,
                                              num_workers=2)
    for i, aa in enumerate(trainloader, 0):
        inputs, labels = aa
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        k = 3
        all_loss[i] = loss.item()
    all_loss = pd.DataFrame(all_loss)
    all_loss = all_loss.sort_values(0)
    print(all_loss[-30:])
    biggest_loss = all_loss[-30:][::-1]
    for index in biggest_loss.index:
        plt.xlabel(test[index][1])
        imshow(test[index][0])


def run_training(criterion, net, optimizer, trainloader, epochs, test_loader, test, data):
    loss_current_epoc = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        total_loss = 0
        j = 0
        for i, dd in enumerate(trainloader, 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = dd
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            if i % 50 == 49:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
            j = i
        out_acc = accuracy_and_confsuion(test_loader, net, test, data)
        loss_current_epoc.append(total_loss / j)

    plt.plot(np.arange(epochs), loss_current_epoc)
    plt.xlabel("epoch num")
    plt.ylabel("training loss")
    plt.show()


def create_sampler(data):
    target = []
    for i, image in enumerate(data):
        target.append(int(image[1]))
    weights = [1 / (7000), 1 / (600), 1 / 2500]
    samples_weight = np.array([weights[t] for t in target])
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)
    return sampler


def imshow(img, save=False, name=""):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
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


def func():
    from skimage import io
    # your path
    col_dir = '/content/img_aug/img_aug/0/*.jpg'

    test = pd.read_pickle("test.pickle")


    testloader = torch.utils.data.DataLoader(test, batch_size=1,
                                             num_workers=2)

    net = SimpleModel()
    net.load("205951270.ckpt")
    accuracy_and_confsuion(testloader, net, test, test)


def get_data(cars, cats, trucks):
    car_images = []
    trucks_images = []
    cats_images = []
    for image in cars:
        image = 2 * (image / 255) - 1
        img_t = torch.tensor(image).float()
        img_t = img_t.permute((2, 0, 1))
        car_images.append((torch.tensor(img_t), 0))
    for image in trucks:
        image = 2 * (image / 255) - 1
        img_t = torch.tensor(image).float()
        img_t = img_t.permute((2, 0, 1))
        trucks_images.append((torch.tensor(img_t), 1))
    for image in cats:
        image = 2 * (image / 255) - 1
        img_t = torch.tensor(image).float()
        img_t = img_t.permute((2, 0, 1))
        cats_images.append((torch.tensor(img_t), 2))

    return car_images, trucks_images, cats_images


if __name__ == "__main__":
    func()
