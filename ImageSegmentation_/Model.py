import torch 
import torchvision.transforms.functional as TF
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, inchannels, outchannels):
        super(DoubleConv, self).__init__()
        self.Conv = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, 3,1,1, bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace= True)
        )
    def forward(self,x):
        x = self.Conv(x)
        return x
    
class UNET(nn.Module):
    def __init__(self, inchannels= 3, outchannels= 1,features = [64,128,256,512]):
        super(UNET, self).__init__()
        self.Downs = nn.ModuleList()
        self.Ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size = 2,stride=2)

        for i in range(len(features)):
            self.Downs.append(DoubleConv(inchannels, features[i]))
            inchannels = features[i]

        for feature in reversed(features):
            self.Ups.append(nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2))

            self.Ups.append(DoubleConv(feature*2, feature))
            inchannels = feature
        self.bottleneck = DoubleConv( features[-1],features[-1]*2)
        self.finalConv = nn.Conv2d(features[0],outchannels, kernel_size=1)
        
    def forward(self, x):
        skip_connections =[]

        for down in self.Downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        x = self.bottleneck(x) 
        skip_connections = skip_connections[::-1]

        for idx in range(0,len(self.Ups),2):
            x = self.Ups[idx](x)
            skip_connection = skip_connections[idx//2] 
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size= skip_connection.shape[2:])

            concat_skipconnection = torch.cat((skip_connection,x),dim=1)
            x = self.Ups[idx+1](concat_skipconnection)

        return self.finalConv(x)

def test():
    x = torch.randn((3,1,160,160))
    model = UNET(inchannels=1, outchannels=1)
    preds = model(x)
    print(x.shape)
    print(preds.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()




