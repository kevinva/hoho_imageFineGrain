import torch
from torchvision import transforms

# IMAGE_DIR = '../data/cub_test/'
IMAGE_DIR = '../data/cub_200_2011/CUB_200_2011/images'

BATCH_SIZE = 16
NUM_EPOCH = 10
LEARNING_RATE = 3e-6
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')