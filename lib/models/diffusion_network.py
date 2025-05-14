import torch
import torch.nn as nn

from lib.losses.contrast_loss import InfoNCE, OcclusionAwareSimilarity
from lib.models.base_network import BaseFeatureExtractor


class DiffusionFeatureExtractor(BaseFeatureExtractor):
    def __init__(self, config, threshold, diffusion_extractor, aggregation_network):
        super(BaseFeatureExtractor, self).__init__()
        assert config.model.backbone == "diffusion", print("Backbone should be Stable Diffusion!")

        self.output_resolution = config.model.output_resolution
        self.loss = InfoNCE()
        self.occlusion_sim = OcclusionAwareSimilarity(threshold=threshold)
        self.sim_distance = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.diffusion_extractor = diffusion_extractor
        self.aggregation_network = aggregation_network
        self.descriptor_size = config.model.descriptor_size

    def forward(self, x):
        with torch.autocast("cuda"):
            feats = self.diffusion_extractor.forward(x)
            # print(f"Dims of 1st feature: {feats[0].shape}\nDims of 2nd feature: {feats[1].shape}")
            diffusion_feats = self.aggregation_network(feats)
            # print(f"Dims after aggregation network: {diffusion_feats.shape}")
            return diffusion_feats
