import argparse
import torch
import torch.optim
import torch.nn
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from models import c3d

# https://github.com/TianzhongSong/3D-ConvNets-for-Action-Recognition/blob/master/train.py

class UCFDataset(Dataset):
   def __init__(self, dataset, labels, num_frames, transforms=None):
      self.dataset = dataset
      self.labels = labels
      self.num_frames = num_frames
      self.transforms = transforms
      self.x_size = 128
      self.y_size = 171
      self.fps = 25
      

   def __len__(self):
      return len(self.labels)

   def read_video(self, video):
      cap = cv2.VideoCapture(video)
      frames = torch.FloatTensor(3, self.num_frames, self.y_size, self.x_size) 

      count, frame_count = 0, 0
      while count < self.num_frames:
         ret, frame = cap.read()
         frame_count += 1

         if not ret:
            break

         if frame_count % self.fps == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.x_size, self.y_size))
            frame = torch.from_numpy(frame)
            frame = frame.permute(2, 0, 1)
            frames[:, count, :, :] = frame 
            count += 1

      return frames

   def __getitem__(self, idx):

      video_file = self.dataset[idx]
      frames = self.read_video(video_file)
      # frames = self.transforms(frames)
      label = int(self.labels[idx])
      return {'frames': frames, 'label': label}


def read_file():
   dataset = {}
   labels = []
   f = open(args.train_path, 'r')
   count = 0

   for line in f:
      line = line.strip('\n')
      line = line.split(' ')
      
      dataset[count] = args.img_path + line[0]
      labels.append(int(line[1]) - 1)
      count += 1

   f.close()
   return dataset, labels


def main():
   # get filenames and labels from text file
   dataset, labels = read_file()

   # load data
   transform = transforms.Compose([transforms.ToTensor()])
   num_frames = 8

   ucf101 = UCFDataset(dataset, labels, args.num_frames, transform)

   test_loader = torch.utils.data.DataLoader(ucf101, batch_size=1, shuffle=True, num_workers=4)

   # get model
   if args.model == 'c3d':
      model = c3d.C3D()
   # elif args.model == 'resnet_3d':
   #    model = resnet_3d.resnet_3d(num_classes, input_shape, drop_rate=args.drop_rate)
   # elif args.model == 'densenet_3d':
   #    model = densenet_3d.densenet_3d(num_classes, input_shape, dropout_rate=args.drop_rate)
   # elif args.model == 'inception_3d':
   #    model = inception_3d.inception_3d(num_classes, input_shape)
   # elif args.model == 'dense_resnet_3d':
   #    model = DenseResNet_3d.dense_resnet_3d(num_classes, input_shape, dropout_rate=args.drop_rate)


   # optimizer
   sgd = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)

   # loss function 
   cross_entropy = torch.nn.CrossEntropyLoss()


   for epoch in range(1, args.epochs+1):
      losses = []
      correct = 0
      count = 0

      for batch_idx, data_point in enumerate(test_loader):
         if count < args.batch_size:
            if count == 0:
               batch = data_point['frames'].clone()
               labels = data_point['label'].clone()
            else:
               batch = torch.cat((batch, data_point['frames']), dim=0)
               labels = torch.cat((labels, data_point['label']), dim=0)
            count += 1
            if batch_idx < len(test_loader.dataset) - 1:
                continue

         # batch created, start training
         model.train().cuda()
         batch = batch.cuda()
         labels = labels.cuda()

         # reset optimizer
         sgd.zero_grad()

         # run through model
         output = model(batch)

         # loss
         loss = cross_entropy(output, labels)
         loss.backward()
         losses.append(loss.item())

         # update weights
         sgd.step()

         # get accuracy
         prediction = output.argmax(dim=1, keepdim=True)
         correct += prediction.eq(labels.view_as(prediction)).sum().item()

         # reset count to 0
         count = 0

      # calculate loss
      train_loss = float(np.mean(losses))
      train_acc = float(100 * correct) / len(test_loader.dataset)

      print(train_loss, train_acc)


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--model', type=str, default='c3d',
                     help='supports resnet_3d, densenet_3d, inception_3d, c3d, dense_resnet_3d')
   
   parser.add_argument('--lr', type=float, default=0.003, help='the initial learning rate')
   parser.add_argument('--batch-size', type=int, default=10)
   parser.add_argument('--img-path', type=str, default='UCF-101/', help='image path')
   parser.add_argument('--train-path', type=str, default='train_file.txt')
   parser.add_argument('--test-path', type=str, default='test_file.txt')
   parser.add_argument('--epochs', type=int, default=10)
   parser.add_argument('--num-frames', type=int, default=8)

   args = parser.parse_args()
   main()
