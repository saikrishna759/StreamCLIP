# https://arxiv.org/pdf/2103.00020.pdf
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from transformers import DistilBertTokenizerFast, DistilBertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # image encoder
        # https://pytorch.org/vision/stable/models.html
        self.image_encoder = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        self.image_encoder.fc = nn.Identity()
        # freeze encoder parameters
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        self.ie_linear_1 = nn.Linear(512,256)
        self.ie_linear_2 = nn.Linear(256,128)

        # text encoder
        # https://huggingface.co/distilbert/distilbert-base-uncased
        self.text_encoder = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
        # freeze encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        self.te_linear_1 = nn.Linear(768,256)
        self.te_linear_2 = nn.Linear(256,128)


    def forward(self, images, texts):

        images = self.image_encoder(images)
        images = F.relu(self.ie_linear_1(images))
        images = F.normalize(self.ie_linear_2(images))

        texts = self.text_encoder(**texts)
        texts = texts.last_hidden_state[:,0,:] # cls token
        texts = F.relu(self.te_linear_1(texts))
        texts = F.normalize(self.te_linear_2(texts))

        cosine_similartiy = images @ texts.T


        return cosine_similartiy
