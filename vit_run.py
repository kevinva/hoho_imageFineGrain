import imp
from urllib.request import urlretrieve
import os
import numpy as np
import torch
from torch import nn, optim
from models.modeling import VisionTransformer, CONFIGS
from data_load import *


def fine_tune(encoder, classifier, train_dataloader, val_dataloader, n_epochs, lr, device):
  encoder.to(device)
  classifier.to(device)

  optimizer_encoder = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.9))
  optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr, betas=(0.5, 0.9))
  criterion = nn.CrossEntropyLoss()

  att_weights = list()

  train_losses, val_losses, val_accs = [], [], []
  for epoch in range(n_epochs):
    encoder.train()
    classifier.train()

    train_loss = 0.
    for batch_i, (img, tgt) in enumerate(train_dataloader):
      img, tgt = img.to(device), tgt.to(device)
      embeddings, att_weights = encoder(img)   # embedding size: [64, 197, 768] (batch_size x (1 + patch的数量) x (16x16x3))

      embedding_cls_token = embeddings[:, 0, :]
      logits = classifier(embedding_cls_token)

      optimizer_encoder.zero_grad()
      optimizer_classifier.zero_grad()
      loss = criterion(logits.squeeze(-1).to(device), tgt)
      loss.backward()
      optimizer_encoder.step()
      optimizer_classifier.step()
      train_loss += loss.item()

    encoder.eval()
    classifier.eval()
    val_loss = 0.
    for batch_i, (img, tgt) in enumerate(val_dataloader):
      img, tgt = img.to(device), tgt.to(device)
      with torch.no_grad():
        embeddings, att_weights = encoder(img)
        embedding_cls_token = embeddings[:, 0, :]
        logits = classifier(embedding_cls_token)
        loss = criterion(logits.squeeze(-1).to(device), tgt)
        val_loss += loss.item()
    
    print('Epoch: {}/{} Val loss: {:.5f}'.format(epoch + 1, n_epochs, val_loss / len(val_dataloader)))
    torch.save(encoder.state_dict(), './model_output/encoder_{}.pt'.format(epoch + 1))
    torch.save(classifier.state_dict(), './model_output/classifier_{}.pt'.format(epoch + 1))
  
  print(f'attn_weight shape: {np.array(att_weights).shape}')
  return encoder, classifier


os.makedirs('model_checkpoints', exist_ok=True)
if not os.path.isfile('model_checkpoints/ViT-B_16-224.npz'):
  urlretrieve('https://storage.googleapis.com/vit_models/imagenet21k+imagenet2012/ViT-B_16-224.npz', 'model_checkpoints/ViT-B_16-224.npz')

config = CONFIGS['ViT-B_16']

# zero_heads: 是否用0初始化网络权重
# vis: 似乎是是否返回注意力权重
model = VisionTransformer(config, num_classes=1000, zero_head=False, img_size=224, vis=True)
model.load_from(np.load('model_checkpoints/ViT-B_16-224.npz'))

encoder = nn.Sequential(model.transformer.embeddings,
                        model.transformer.encoder)
ViT_embed_dim = 768
n_classes = len(class2idx)
classifier = nn.Linear(ViT_embed_dim, n_classes)

if os.path.exists('model_output/encoder_1.pt'):
    encoder.load_state_dict(torch.load('model_output/encoder_1.pt'))
if os.path.exists('model_output/classifier_1.pt'):
    classifier.load_state_dict(torch.load('model_output/classifier_1.pt'))


encoder, classifier = fine_tune(encoder, 
                                classifier,
                                train_dataloader,
                                val_dataloader,
                                NUM_EPOCH,
                                LEARNING_RATE,
                                DEVICE)