from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from sklearn.manifold import TSNE


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(5, 5), stride=1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.dropout1 = nn.Dropout2d(0.4)
        self.dropout2 = nn.Dropout2d(0.3)
        self.dropout3 = nn.Dropout2d(0.2)
        self.fc1 = nn.Linear(128 * 9, 256)
        self.fc2 = nn.Linear(256, 10)


    def forward(self, x, get_features=True):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(self.bn2(x))

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout3(self.bn3(x))

        x = torch.flatten(x, 1)
        feat = self.fc1(x)

        x = F.relu(feat)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output, feat


if __name__ == '__main__':
    model = Net()
    model.load_state_dict(torch.load('mnist_model_full.pt'))

    test_dataset = datasets.MNIST('../data', train=False,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))

    with torch.no_grad():   # For the inference step, gradient is not computed
        mistake_ims = []
        mistakes = []
        model.eval()
        kernels = list(model.children())[0].weight.data[:9, 0]
        confusion = np.zeros((10, 10))
        class_counts = np.zeros(10)
        features = np.zeros((len(test_dataset), 256))
        targets = []
        i = 0

        for data, target in test_dataset:
            targets.append(target)
            output, features[i] = model(data.unsqueeze(0))
            pred = int(output.argmax(dim=1, keepdim=True)[0])

            if pred != target and len(mistakes) < 9:
                mistake_ims.append(data[0])
                mistakes.append((pred, target))

            confusion[target, pred] += 1
            class_counts[target] += 1
            i += 1

        #feats_embedded = TSNE(n_components=2).fit_transform(features)

        n_train = [60000 // (2 ** i) for i in [0, 1, 2, 3, 4]]
        train_accs = [99.43, 99.40, 99.28, 99.23, 99.]
        test_accs = [99.5, 99.44, 99.25, 99.19, 98.65]
        plt.figure(figsize=(8, 4))
        plt.loglog(n_train, 1 - np.array(train_accs) / 100., label='train')
        plt.loglog(n_train, 1 - np.array(test_accs) / 100., label='test')
        plt.xlabel('number of training examples')
        plt.ylabel('zero-one error')
        plt.xticks(n_train, n_train)
        plt.yticks([0.005, 0.01, 0.015])
        plt.legend()
        plt.savefig('subset_training.png')
        plt.close()

        fig, axs = plt.subplots(3, 3)
        for i, im in enumerate(mistake_ims):
            axs[i // 3, i % 3].imshow(im, cmap='gray')
            axs[i // 3, i % 3].axis('off')
            axs[i // 3, i % 3].set_title(f'Pred: {mistakes[i][0]}, Actual: {mistakes[i][1]}')
        plt.savefig('mistake_examples.png')
        plt.close()

        fig, axs = plt.subplots(3, 3)
        for i, im in enumerate(kernels):
            axs[i // 3, i % 3].imshow(im, cmap='gray')
            axs[i // 3, i % 3].axis('off')
            axs[i // 3, i % 3].set_title(f'Kernel {i + 1}')
        plt.savefig('first_layer_kernels.png')
        plt.close()

        confusion = [[('%d' % p) for p in row] for row in confusion]
        fig, axs = plt.subplots(1, 1)
        axs.table(cellText=confusion, cellLoc='center', loc='center')
        axs.axis('off')
        axs.set_ylabel('True Class')
        axs.set_title('Predicted Class')
        plt.savefig('confusion_matrix.png')
        plt.close()

        plt.figure(figsize=(8, 8))
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                  'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        for i, c in enumerate(colors):
            fes = feats_embedded[np.where(np.array(targets) == i)]
            plt.scatter(fes[:, 0], fes[:, 1], color=colors[i], s=1, label=str(i))
        plt.legend()
        plt.savefig('tSNE.png')
        plt.close()

        plt.rcParams["font.family"] = "serif"
        fig, axs = plt.subplots(4, 9)
        idxs = np.random.choice(len(test_dataset), size=4, replace=False)
        for i, idx in enumerate(idxs):
            sq_dists = np.sum((feats_embedded - feats_embedded[idx]) ** 2, axis=1)
            neighbors = np.argsort(sq_dists)[1:9]
            axs[i, 0].imshow(test_dataset[idx][0][0], cmap='gray')
            axs[i, 0].axis('off')
            for j in range(8):
                axs[i, j + 1].imshow(test_dataset[neighbors[j]][0][0], cmap='gray')
                axs[i, j + 1].axis('off')
        axs[0, 0].set_title('I_0')
        for j in range(1, 9):
            axs[0, j].set_title('I_' + str(j))
        plt.savefig('neighbors.png')
        plt.close()
