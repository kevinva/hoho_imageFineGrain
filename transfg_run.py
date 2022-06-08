import torch
from torch import cos_, nn, optim
import torch.nn.functional as F

from models.modeling import Block, LayerNorm

import time


def fetch_part_attention(vit_features, att_mat_list):
    # att_mat_list为 12(层) x [batch_size, 12(head), 197, 197]
    print(f'feature size: {vit_features.size()}')
    print(f'att mat size: {att_mat_list[0].size()}')
    
    joint_att_mat = att_mat_list[0]   # [batch_size, 12(head_size), 197, 197]
    for n in range(1, len(att_mat_list)):
        joint_att_mat = torch.matmul(att_mat_list[n], joint_att_mat)

    print(f'1. joint_att_mat size: {joint_att_mat.size()}')
    joint_att_mat = joint_att_mat[:, :, 0, 1:]  # [batch_size, head_size, 196]，找CLS_token attent到其他token的注意力
    print(f'2. joint_att_mat size: {joint_att_mat.size()}')
    return joint_att_mat.max(2)[1]

def fetch_part_features(vit_features, att_index):
    att_index = att_index + 1  # CLS_token的index为0，所以要+1以从图像token里选取
    parts = []
    batch_num, head_num = att_index.shape
    for b in range(batch_num):
        parts.append(vit_features[b, att_index[b, :], :])
    parts = torch.stack(parts).squeeze(1)
    cls_feature = vit_features[:, 0, :].unsqueeze(1)
    feature_selected = torch.cat((cls_feature, parts), dim=1)
    return feature_selected


def contrastive_loss(features, targets):
    batch_num, _ = features.shape
    features = F.normalize(features)
    cos_mat = features.mm(features.t())
    pos_label_mat = targets.mm(targets.t())  # targets为one-hot矩阵
    neg_label_mat = 1 - pos_label_mat
    pos_cos_mat = 1 - cos_mat
    neg_cos_mat = cos_mat - 0.4
    neg_cos_mat[neg_cos_mat < 0] = 0
    loss = (pos_cos_mat * pos_label_mat).sum() + (neg_cos_mat * neg_label_mat).sum()
    loss /= (batch_num * batch_num)
    return loss


class PartLayer(nn.Module):

    def __init__(self, vit_config):
        self.part_transformer = Block(vit_config)
        self.part_norm = LayerNorm(vit_config.hidden_size, eps=1e-6)

    def forward(self, vit_features, att_weight_list):
        att_part_index = fetch_part_attention(vit_features, att_weight_list)
        part_feature = fetch_part_features(vit_features, att_part_index)
        part_states, part_attention_weights = self.part_transformer(part_feature)
        part_states = self.part_norm(part_states)

        return part_states, part_attention_weights


def transfg_fine_tune(encoder, partlayer, classifier, train_dataloader, val_dataloader, n_epochs, lr, device):
    encoder.to(device)
    partlayer.to(device)
    classifier.to(device)

    optimizer_encoder = optim.Adam(encoder.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_partlayer = optim.Adam(partlayer.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=lr, betas=(0.5, 0.9))
    cross_criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    for epoch in range(n_epochs):
        encoder.train()
        partlayer.train()
        classifier.train()

        train_loss = 0.
        for batch_i, (img_x, target) in enumerate(train_dataloader):
            img_x, target = img_x.to(device), target.to(device)
            features, att_weight_list = encoder(img_x)
            part_features, part_att_weights = partlayer(features, att_weight_list)

            cls_features = part_features[:, 0, :]
            logits = classifier(cls_features)

            optimizer_encoder.zero_grad()
            optimizer_partlayer.zero_grad()
            optimizer_classifier.zero_grad()
            loss = cross_criterion(logits.squeeze(-1).to(device), target) + contrastive_loss(cls_features, target)
            loss.backward()
            optimizer_encoder.step()
            optimizer_partlayer.step()
            optimizer_classifier.step()

            train_loss += loss.item()

        encoder.eval()
        partlayer.eval()
        classifier.eval()
        val_loss = 0.
        for batch_i, (img, tgt) in enumerate(val_dataloader):
            img_x, target = img_x.to(device), target.to(device)
            with torch.no_grad():
                features, att_weight_list = encoder(img_x)
                part_features, part_att_weights = partlayer(features, att_weight_list)

                cls_features = part_features[:, 0, :]
                logits = classifier(cls_features)
                loss = cross_criterion(logits.squeeze(-1).to(device), target) + contrastive_loss(cls_features, target)
                val_loss += loss.item()
        
        print('Epoch: {}/{}, Val loss: {:.5f}, elpase: {:.3f}s'.format(epoch + 1, n_epochs, val_loss / len(val_dataloader), time.time() - start_time))
        torch.save(encoder.state_dict(), './model_output/transfg_encoder_{}.pt'.format(epoch + 1))
        torch.save(partlayer.state_dict(), './model_output/transfg_partlayer_{}.pt'.format(epoch + 1))
        torch.save(classifier.state_dict(), './model_output/transfg_classifier_{}.pt'.format(epoch + 1))


        print(f'att_weights len: {len(part_att_weights)}')
        for item in part_att_weights:
            print(f'    item size: {item.size()}')


    return encoder, partlayer, classifier

