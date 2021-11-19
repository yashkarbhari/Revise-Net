#reverse attention with edge attention
class RAE(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel):
        super(RAE, self).__init__()
        self.convert1 = nn.Conv2d(in_channel, mid_channel, 1)
        self.convert2 = nn.Conv2d((3*mid_channel)//2, out_channel, 1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.BatchNorm2d(out_channel), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.channel = out_channel
        self.convert3 = nn.Conv2d(1024, 768, 1)
        self.convert4 = nn.Conv2d(512,384,1)
        self.convert5 = nn.Conv2d(256,192,1)
        #self.conva = nn.Conv2d(772, 256, 1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, y, z): #y is prediction and x is feature map of previous unit and z is current skip

        a = torch.sigmoid(-y)
        x = self.convert1(x)
        x = F.interpolate(x, z.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, z), dim=1)
        if (x.shape[1]==1024):
            x = self.convert3(x)
        elif (x.shape[1]==512 and x.shape[2]==56):
            x = self.convert4(x)
        elif (x.shape[1]==256 and x.shape[2]==112):
            x = self.convert5(x)
        elif (x.shape[1]==512 and x.shape[2]==64):
            x = self.convert4(x)
        elif (x.shape[1]==256 and x.shape[2]==128):
            x = self.convert5(x)
        elif (x.shape[1]==512 and x.shape[2]==80):
            x = self.convert4(x)
        elif (x.shape[1]==256 and x.shape[2]==160):
            x = self.convert5(x)
        x = self.relu(self.bn(self.convert2(x)))

        '''
        if(a.shape[1]==772):
          a = self.conva(a)
        if(x.shape[3]==a.shape[3]):
          #x = a.expand(-1, self.channel, -1, -1).mul(x)
          x = a.mul(x)
        else:
          x = self.up(x)
          #x = a.expand(-1, self.channel, -1, -1).mul(x)
          x = a.mul(x)
        '''
        y = y + self.convs(x)

        return y
