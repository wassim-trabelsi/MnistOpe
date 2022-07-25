import torch.nn as nn
import torch.nn.functional as F
import torch


class SumNet(nn.Module):
    def __init__(self):
        super(SumNet, self).__init__()
        # Create an embedding (Same as in the simple classifier but removed the last layer) (see classifier/models.py)
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)

        self.concat_layer1 = nn.Linear(256, 256)
        self.concat_layer2 = nn.Linear(256, 19)

    def forward_one_image(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

    def forward(self, im1, im2):
        embedding_im1 = self.forward_one_image(im1)
        embedding_im2 = self.forward_one_image(im2)
        # Concatenate the embeddings
        embedding = torch.cat((embedding_im1, embedding_im2), 1)
        # Pass the embedding through the last layer
        output = F.relu(self.concat_layer1(embedding))
        output = F.log_softmax(self.concat_layer2(output), dim=1)
        return output
