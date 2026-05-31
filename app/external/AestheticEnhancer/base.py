import torch
from torchvision import transforms
import numpy as np

factors = ['saturation', 'brightness', 'tint', 'temperature', 'contrast']
weight_0 = 1
epochs = 25
learning_rate = 0.001
device = 'cuda:1'
save_dir = 'results/norm_40K_balanced_small-range'

# Model
model = {
    'n_factors': len(factors),
    'activation': 'tanh',
    'backbone': 'resnet18'
}

# Dataset
img_transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

factors_coefs = {
    'saturation': 43,
    'brightness': 43,
    'tint': 30,
    'temperature': 30,
    'contrast': 43,
}

def get_label_normalizer(factors, factors_coefs):
    # Convert dict to list arcording to factors order
    factors_coefs = [factors_coefs[factor] for factor in factors]
    factors_coefs = np.array(factors_coefs, dtype=np.float16)

    class Normalizer:
        def __init__(self, factors_coefs):
            self.factors_coefs = factors_coefs

        def transform(self, labels):
            return labels / self.factors_coefs

        def inverse_transform(self, norm_labels):
            return norm_labels * self.factors_coefs
        
    return Normalizer(factors_coefs)

base_data = {
    'factors': factors,
    'img_transform': img_transform,
    'label_normalizer': get_label_normalizer(factors, factors_coefs), # If no norm --> set to None
    'weight_0': weight_0,
}

train_data, val_data, test_data = base_data.copy(), base_data.copy(), base_data.copy()

train_data.update({
    'gt_file': './data/40K_balanced_small-range/csv/final_train.csv',
    'img_dir': './data/40K_balanced_small-range/images/train',
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 4,
})

val_data.update({
    'gt_file': './data/40K_balanced_small-range/csv/final_val.csv',
    'img_dir': './data/40K_balanced_small-range/images/val',
    'batch_size': 16,
    'shuffle': False,
    'num_workers': 4,
})

test_data.update({
    'gt_file': './data/40K_balanced_small-range/csv/final_test.csv',
    'img_dir': './data/40K_balanced_small-range/images/test',
    'batch_size': 16,
    'shuffle': False,
    'num_workers': 4,
})