import torch
from torch import nn
from ..vision_transformer.modeling import VisionTransformer, CONFIGS


class ResBlock(nn.Module):
    def __init__(
            self, feature_dims, projection_dim, stride, downsample_rate=2
    ):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(feature_dims, feature_dims // downsample_rate, kernel_size=1, bias=False),
            nn.ReLU(),
        )
        self.conv1.apply(self.zero_init)

        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_dims // downsample_rate, projection_dim, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.ReLU(),
        )
        self.conv2.apply(self.zero_init)

        self.conv3 = nn.Conv2d(projection_dim, projection_dim, kernel_size=1, bias=False)
        self.conv3.apply(self.zero_init)

        self.short_cut = nn.Conv2d(feature_dims, projection_dim, kernel_size=3, stride=stride, padding=1, bias=False)

    def zero_init(self, m):
        if isinstance(m, nn.Conv2d):
            m.weight.data.fill_(0.)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out) + self.short_cut(x)
        return out

class FusionModule(nn.Module):
    def __init__(self, feature_dim, n_features, transformer_hidden_size):
        super(FusionModule, self).__init__()
        self.feature_dim = feature_dim
        self.n_features = n_features
        self.transformer_hidden_size = transformer_hidden_size
        self.weighting = nn.Sequential(
            nn.Flatten(),
            nn.Linear(transformer_hidden_size * n_features, n_features, bias=True)           # Modified to allow weighting by tranformer output instead of extracted features
        )
        self.weighting.apply(self.init)

    def init(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.fill_(0.)
            m.bias.data.fill_(1.)

    def forward(self, x, transformer_outputs):
        # Finding the weights
        weight = nn.functional.softmax(self.weighting(transformer_outputs), dim=1)          # Reploced weighting by tranformer output instead of extracted features
        out = None
        # Averaging with the weights from MLP
        for idx in range(self.n_features):
            start, end = idx * self.feature_dim, (idx + 1) * self.feature_dim
            if out is None:
                out = torch.einsum('b, bchw-> bchw', weight[:, idx], x[:, start:end, :, :])
            else:
                out = out + torch.einsum('b, bchw-> bchw', weight[:, idx], x[:, start:end, :, :])
        return out


class AggregationNetwork(nn.Module):
    def __init__(
            self, 
            feature_dims, 
            device,
            stride=2,
            descriptor_size=16,
    ):
        super().__init__()
        self.extractors = nn.ModuleList()
        self.feature_dims = feature_dims
        self.device = device

        for l in range(len(self.feature_dims)):
            extractor = ResBlock(
                self.feature_dims[l],
                descriptor_size,
                stride
            )
            self.extractors.append(extractor)
        self.extractors = self.extractors.to(device)

        # Vision Transformer
        self.vision_transformer_config = CONFIGS['ViT-Diffusion']
        self.img_size = 32          # Height and width sizes of features
        self.embeddings_in_channels = 16
        self.num_classes = 10       # Number of classes to classify to (will have to change)
        self.vision_transformer = VisionTransformer(self.vision_transformer_config, self.embeddings_in_channels, 
                                                        img_size = self.img_size,zero_head = True, 
                                                            num_classes = self.num_classes, vis = False) 
        # vis = True -> returns the weights of all attention layers and heads

        self.transformer_hidden_size = 64
        self.fusion = FusionModule(descriptor_size, len(self.feature_dims), self.transformer_hidden_size)
        self.fusion = self.fusion.to(device)

    def forward(self, batch):
        extracted_features = []
        transformer_outputs = []
        
        for i in range(len(self.feature_dims)):
            extractor = self.extractors[i]
            feats = batch[i]
            extracted_feature = extractor(feats)
            extracted_features.append(extracted_feature)

            # Adding transformer here
            transformer_output, attn_weights = self.vision_transformer(extracted_feature)
            transformer_outputs.append(transformer_output)
        
        extracted_features = torch.cat(extracted_features, dim=1)
        transformer_outputs = torch.stack(transformer_outputs, dim = 1)
        
        fused_feature = self.fusion(extracted_features, transformer_outputs)
        
        return fused_feature
