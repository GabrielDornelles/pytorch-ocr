import torch
from torch import nn
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
import config

class EchidNet(nn.Module):

    def __init__(self, n_classes):
        super().__init__()

        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=4, stride=1),
            nn.ReLU6(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=10, kernel_size=4, stride=1),
            nn.ReLU6(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=10, out_channels=60, kernel_size=5, stride=1),
            nn.ReLU6()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=60, out_features=40),
            nn.ReLU6(),
            nn.Linear(in_features=40, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs
    
class OcrModel(nn.Module):
    def __init__(self, num_chars):
        super(OcrModel, self).__init__()

        self.conv_1 = nn.Conv2d(3, 256, kernel_size=(3, 5), padding=(1, 1)) # change these kernels depending on your input shape ratio, 
        # here is 75x25, 75/25=3, 3/2= 1.5, should be your kernel ratio, 3*1.5=4.5 > 5, thus 3x5
        self.pool_1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv_2 = nn.Conv2d(256, 64, kernel_size=(3, 5), padding=(1, 1))
        self.pool_2 = nn.MaxPool2d(kernel_size=(2, 2))

        self.linear_1 = nn.Linear(384 , 128) #if you changed the kernels above then matrices won't match, change the input of that layer to the right shape. there's debugs on x.shape on forward pass.
        self.drop_1 = nn.Dropout(0.2)
        self.lstm = nn.GRU(128, 128, bidirectional=True, num_layers=4, dropout=0.25, batch_first=True) #before, 2 layers 64 outputs bidirectional
        self.output = nn.Linear(256, num_chars + 1)
        
        # trying to use efficientnet as the cnn
        # self.network = EfficientNet.from_pretrained('efficientnet-b0', num_classes=34)
        # self.features = model.extract_features(img)
        # self.network._fc = nn.Linear(self.network._fc.in_features, 64)
        # self.linear_2 = nn.Linear(1280,64)
        # self.drop_1 = nn.Dropout(0.2)
        # self.lstm = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        # self.output = nn.Linear(64, num_chars + 1

        # self.lstm = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25, batch_first=True)
        # self.output = nn.Linear(64, num_chars + 1)

    def forward(self, images, targets=None):
        # bs, _, _, _ = images.size()
        # #x = images.view(bs, images.size(1), -1)
        # #print(x.shape)
        # #images = images[None, :, :]
        # #print(f"[INFO]: Image shape: {images.shape}")
        # #x = self.network(images) # torch is bs, 64, should be torch.Size([bs, img_width after network processment, 64])
        # net = self.network
        # #x = net.extract_features(images)
        # #print(x.shape) # -> torch.Size([32, 1280, 1, 3])
        # #x = self.network(images)
        # x = net.extract_features(images)
        # #print(x.shape)
        # #x = self.drop_1(x)
        # x = x.permute(0, 3, 1, 2)
        # x = x.view(bs, x.size(1), -1)
        # #x = F.relu(self.linear_2(x))
        # #print(x.shape)
        # #print(f"[INFO]: extracted features: {x.shape}")
        # #print(x.shape)
        # #x = x.view(bs, 22, 64)
        # #print(x.shape)

        # #print(f"[INFO]: Shape before LSTM layer: {x.shape}")
        # x,_ = self.lstm(x)
        # #print(x.shape)
        # x = self.output(x)
        # #print(x.shape)
        # x = x.permute(1, 0, 2)

        #print(images.shape)
        bs, _, _, _ = images.size()
        #print(images.shape)
        x = F.relu(self.conv_1(images))
        #print(x.shape)
        x = self.pool_1(x)
        #print(x.shape)
        x = F.relu(self.conv_2(x))
        #print(x.shape)
        x = self.pool_2(x)
       # print(x.shape) # -> ([32, 64, 6, 15])
        #print(f"X dps do pool: {x.shape}")
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        #print(f"Shape before Linear layer: {x.shape}")
        x = F.relu(self.linear_1(x))
         # 22, 15, 64]
        x = self.drop_1(x)
        #print(f"[INFO]: Shape before LSTM layer: {x.shape}") #its bs, ?, number of filters, where ? might be something of image matrix
        x, _ = self.lstm(x)
        x = self.output(x)
        x = x.permute(1, 0, 2)

        if targets is not None: #I just need to know how big its here to pass target_length fill value
            log_probs = F.log_softmax(x, 2)
            #print(f"logs size: {targets}")
            input_lengths = torch.full(
                size=(bs,), fill_value=log_probs.size(0), dtype=torch.int32
            )
            #print(f"logs size: {targets.size(0)}")

            target_lengths = torch.full( # the problem about length is here, can trick by passing filling what is smaller than max length, then just remove it
                size=(bs,), fill_value=targets.size(1), dtype=torch.int32
            )
            loss = nn.CTCLoss(blank=0)(
                log_probs, targets, input_lengths, target_lengths
            )
            return x, loss

        return x, None


if __name__ == "__main__":
    pass
    #debug shapes
    #cm = OcrModel(33)
    #img = torch.rand((1, 3, 50, 200))
    #x, _ = cm(img, torch.rand((1, 5)))
