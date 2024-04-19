"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
import torch
from torchvision import models

from .JCAttetnionModule.av_crossatten import DCNLayer
from .JCAttetnionModule.audguide_att import BottomUpExtract

class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input


class VideoModel(nn.Module):
    def __init__(self, num_channels=3):
        super(VideoModel, self).__init__()
        self.r2plus1d = models.video.r2plus1d_18(pretrained=True)
        
        print("num_channels in VideoModel: ", num_channels)
        self.r2plus1d.fc = nn.Sequential(nn.Dropout(0.0),
                                         nn.Linear(in_features=self.r2plus1d.fc.in_features, out_features=17))
        if num_channels == 4:
            new_first_layer = nn.Conv3d(in_channels=4,
                                        out_channels=self.r2plus1d.stem[0].out_channels,
                                        kernel_size=self.r2plus1d.stem[0].kernel_size,
                                        stride=self.r2plus1d.stem[0].stride,
                                        padding=self.r2plus1d.stem[0].padding,
                                        bias=False)
            # copy pre-trained weights for first 3 channels
            new_first_layer.weight.data[:, 0:3] = self.r2plus1d.stem[0].weight.data
            self.r2plus1d.stem[0] = new_first_layer
        self.modes = ["clip"]

    def forward(self, x):
        print("Video forward x.shape", x.shape)
        return self.r2plus1d(x)


class AudioModel(nn.Module):
    def __init__(self, pretrained=False):
        super(AudioModel, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Sequential(nn.Dropout(0.0),
                                       nn.Linear(in_features=self.resnet.fc.in_features, out_features=17))

        old_layer = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(1, out_channels=self.resnet.conv1.out_channels,
                                      kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if pretrained == True:
            self.resnet.conv1.weight.data.copy_(torch.mean(old_layer.weight.data, dim=1, keepdim=True)) # mean channel

        self.modes = ["audio_features"]

    def forward(self, x):
        print("Audio forward x.shape", x.shape)
        return self.resnet(x)


class TwoStreamAuralVisualModel(nn.Module):
    def __init__(self, num_channels=3, audio_pretrained=False):
        super(TwoStreamAuralVisualModel, self).__init__()
        self.audio_model = AudioModel(pretrained=audio_pretrained)
        self.video_model = VideoModel(num_channels=num_channels)
        self.fc = nn.Sequential(nn.Dropout(0.0),
                                          nn.Linear(in_features=self.audio_model.resnet.fc._modules['1'].in_features +
                                                                self.video_model.r2plus1d.fc._modules['1'].in_features,
                                                    out_features=17))
        
        new_first_layer = nn.Conv3d(in_channels=3,
					out_channels=self.video_model.r2plus1d.stem[0].out_channels,
					kernel_size=self.video_model.r2plus1d.stem[0].kernel_size,
					stride=self.video_model.r2plus1d.stem[0].stride,
					padding=self.video_model.r2plus1d.stem[0].padding,
					bias=False)
        
        new_first_layer.weight.data = self.video_model.r2plus1d.stem[0].weight.data[:, 0:3]
        self.video_model.r2plus1d.stem[0] = new_first_layer        
        
        self.modes = ['clip', 'audio_features']
        self.audio_model.resnet.fc = Dummy()
        self.video_model.r2plus1d.fc = Dummy()
        # self.video_model.r2plus1d.avgpool = Dummy()
        self.CAM = CAM().cuda()

    def forward(self, clip, audio):
        # audio = x['audio_features']
        # clip = x['clip']
        
        print("clip shape : ", clip.shape)
        print("audio shape : ", audio.shape)
        
        audio_model_features = self.audio_model(audio)
        
        # for i in range(clip.shape[0]):
        #     video_model_features = self.video_model(clip[i,:,:,:,:,:])
        video_model_features = self.video_model(clip)
        
        print("audio_model_features.shape : ", audio_model_features.shape)
        print("video_model_features.shape : ", video_model_features.shape)
        
        features = torch.cat([audio_model_features, video_model_features], dim=1)
        print("features.shape : ", features.shape)
        out = self.fc(features)
        
        print(f"audio_model_features.shape: {audio_model_features.shape}, video_model_features.shape: {video_model_features.shape}")
        # audio_shape, video_shape = [8,512], [8,512]
        
        # out = self.CAM(audio_model_features, video_model_features)
        
        return audio_model_features, video_model_features, out



class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()
        self.coattn = DCNLayer(512, 512, 1, 0.6)

        self.video_attn = BottomUpExtract(512, 512)
        self.vregressor = nn.Sequential(nn.Linear(512, 128),
                                        nn.ReLU(inplace=True),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))

     

        self.aregressor = nn.Sequential(nn.Linear(512, 128),
                                        nn.ReLU(inplace=True),
                                     nn.Dropout(0.6),
                                 nn.Linear(128, 1))

        self.init_weights()

    def init_weights(net, init_type='xavier', init_gain=1):

        if torch.cuda.is_available():
            net.cuda()

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.uniform_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_uniform_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_uniform_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>


    def forward(self, f1_norm, f2_norm):
        video = F.normalize(f2_norm, dim=-1)
        audio = F.normalize(f1_norm, dim=-1)
      
       
        video = self.video_attn(video, audio)
        
        video, audio = self.coattn(video, audio)

        audiovisualfeatures = torch.cat((video, audio), -1)
  
        vouts = self.vregressor(audiovisualfeatures) #.transpose(0,1))
        aouts = self.aregressor(audiovisualfeatures) #.transpose(0,1))

        return vouts.squeeze(2), aouts.squeeze(2)  #final_aud_feat.transpose(1,2), final_vis_feat.transpose(1,2)
