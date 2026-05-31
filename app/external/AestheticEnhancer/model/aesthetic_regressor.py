import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import vision_transformer
import timm

def get_backbone(name, unfreeze_layers_of_vit, img_size):
    print(f"Initializing {name} model...")
    
    if name == 'resnet18': # 11.7M params
        return models.resnet18(pretrained=True)
    elif name == 'resnet50':
        return models.resnet50(pretrained=True)
    elif name == 'mobilenetv3_large': # 5.4M params
        return models.mobilenet_v3_large(pretrained=True)
    elif name == 'mobilenetv3_small': # 2.5M params
        return models.mobilenet_v3_small(pretrained=True)
    elif name == 'vit_b_16':  # 86M params, unfreeze 1 layer = 7M params
        
        model = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=img_size)

        # Free all layers
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze some top encoder layers
        total_encoder_layers = len(model.blocks)

        for i in range(total_encoder_layers - 1, total_encoder_layers - 1 - unfreeze_layers_of_vit, -1):
            print(f'Unfreeze encoder layer {i}')
            for name, param in model.blocks[i].named_parameters():
                param.requires_grad = True
                
        return model
    else:
        raise ValueError(f'Backbone "{name}" not supported, please add backbone in func get_backbone')

def get_activation(name):
    if name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError(f'Function "{name}" not supported, please add activation in func get_activation')

class AestheticRegressor(nn.Module):
    def __init__(self, **kwargs):
        super(AestheticRegressor, self).__init__()
        
        # Extract parameters from kwargs with default values
        n_factors = kwargs.get('n_factors', 5)
        activation = kwargs.get('activation', None)
        backbone = kwargs.get('backbone', 'resnet18')
        unfreeze_layers_of_vit = kwargs.get('unfreeze_layers_of_vit', 1)  # Default value set here
        img_size = kwargs.get('img_size', 640)
        
        # Backbone
        self.backbone = get_backbone(backbone, unfreeze_layers_of_vit, img_size)
        print(f"Number of trainable parameters: {sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)}")
        
        # Regression head
        if backbone in ['resnet18', 'resnet50']:
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, n_factors)
        elif backbone == 'mobilenetv3_large':
            num_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(num_features, n_factors)
        elif backbone == 'mobilenetv3_small':
            num_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(num_features, n_factors)
        elif backbone == 'vit_b_16':
            # Correctly access the classification head
            num_features = self.backbone.head.in_features
            self.backbone.head = nn.Linear(num_features, n_factors)
        
        self.activation = get_activation(activation) if activation else None

    def forward(self, x):
        output = self.backbone(x)
        
        # VIT-B/16 returns a tuple, so we need to extract the first element
        if isinstance(output, tuple):
            output = output[0]

        if self.activation:
            output = self.activation(output)

        return output

if __name__ == '__main__':
    model = AestheticRegressor(n_factors=5, activation='tanh', backbone='vit_b_16')
