"""One anti-spoofing model: loads a weights file once, then classifies face patches."""
import os

import torch
import torch.nn.functional as F

from liveness_service.ml.networks import MODEL_MAPPING
from liveness_service.ml.utils import get_kernel, parse_model_name


class AntiSpoofModel:
    def __init__(self, model_path, device):
        # the filename carries the config, e.g. 2.7_80x80_MiniFASNetV2.pth -> scale 2.7, 80x80 input, V2
        self.height, self.width, model_type, self.scale = parse_model_name(os.path.basename(model_path))
        self.device = device
        self.model = MODEL_MAPPING[model_type](conv6_kernel=get_kernel(self.height, self.width)).to(device)

        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        # strip DataParallel's 'module.' prefix
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        # checkpoints from training hold the whole MultiFTNet: keep only its 'model.' classifier, drop FTGenerator
        if any(k.startswith('model.') for k in state_dict):
            state_dict = {k[6:]: v for k, v in state_dict.items() if k.startswith('model.')}
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict(self, patch):
        """patch: BGR uint8 image of size (self.height, self.width). Returns 3 class probabilities; class 1 = real."""
        # HxWxC -> CxHxW. Pixels stay 0-255 BGR, as in training (its ToTensor has the /255 removed).
        x = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            return F.softmax(self.model(x), dim=1)[0].cpu().numpy()
