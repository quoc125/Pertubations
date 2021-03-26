import torch
import torch.nn as nn 
import torch.nn.functional as F

# class C3D(nn.Module):
#    def __init__(self):
#       super(C3D, self).__init__()

#       # define model layout
#       self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1))
#       self.max_pool_1 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
      
#       self.conv2 = nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
#       self.max_pool_2 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

#       self.conv3 = nn.Conv3d(in_channels=128, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 1, 1))
#       self.max_pool_3 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

#       self.conv4 = nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
#       self.max_pool_4 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))

#       self.conv5 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=(3, 3, 3), stride=(1, 1, 1))
#       self.max_pool_5 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))

#       self.dropout = nn.Dropout3d(p=0.5)

#       self.linear_1 = nn.Linear(in_features=123123, out_features=2048)
#       self.linear_2 = nn.Linear(in_features=2048, out_features=2048)
#       self.linear_3 = nn.Linear(in_features=2048, out_features=101)

#       self.softmax = nn.Softmax(dim=1)




#    def forward(self, x):

#       print(x.shape)

#       x = self.conv1(x)
#       x = F.relu(x)
#       x = self.max_pool_1(x)

#       print(x.shape)

#       x = self.conv2(x)
#       x = F.relu(x)
#       x = self.max_pool_2(x)
      
#       print(x.shape)

#       x = self.conv3(x)
#       x = F.relu(x)
#       x = self.max_pool_3(x)

#       print(x.shape)

#       x = self.conv4(x)
#       x = F.relu(x)
#       x = self.max_pool4(x)

#       x = self.conv5(x)
#       x = F.relu(x)
#       x = self.max_pool5(x)

#       x = torch.flatten(x, start_dim=1)

#       print(x.shape)

#       x = self.linear_1(x)
#       x = self.dropout(x)
#       x = self.linear_2(x)
#       x = self.dropout(x)
#       x = self.linear_3(x)
#       x = self.softmax(x)

#       return x

class C3D(nn.Module):
   """
   The C3D network as described in [1].
   https://github.com/DavideA/c3d-pytorch/blob/master/C3D_model.py
   """

   def __init__(self):
      super(C3D, self).__init__()

      self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
      self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

      self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
      self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

      self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
      self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
      self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

      self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
      self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
      self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

      self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
      self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
      self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1))

      self.fc6 = nn.Linear(15360, 4096)
      self.fc7 = nn.Linear(4096, 4096)
      self.fc8 = nn.Linear(4096, 101)

      self.dropout = nn.Dropout(p=0.5)

      self.relu = nn.ReLU()
      self.softmax = nn.Softmax()

   def forward(self, x):

      h = self.relu(self.conv1(x))
      h = self.pool1(h)

      h = self.relu(self.conv2(h))
      h = self.pool2(h)

      h = self.relu(self.conv3a(h))
      h = self.relu(self.conv3b(h))
      h = self.pool3(h)

      h = self.relu(self.conv4a(h))
      h = self.relu(self.conv4b(h))
      h = self.pool4(h)

      h = self.relu(self.conv5a(h))
      h = self.relu(self.conv5b(h))
      h = self.pool5(h)

      # print(h.shape)
      h = torch.flatten(h, start_dim=1)
      h = self.relu(self.fc6(h))
      h = self.dropout(h)
      h = self.relu(self.fc7(h))
      h = self.dropout(h)

      logits = self.fc8(h)
      probs = self.softmax(logits)

      return probs
