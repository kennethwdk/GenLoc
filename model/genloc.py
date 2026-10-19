import torch
import torch.nn as nn
import lightning as L
import numpy as np
from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
from torch.optim.lr_scheduler import StepLR

from dataset.transforms import transform_preds

class GenLocPLModel(L.LightningModule):
    def __init__(self, model, learning_rate=1e-2):
        super().__init__()
        self.model = model
        self.lr = learning_rate

    def training_step(self, batch, batch_idx):
        image, heatmap, keypoint, mask, meta = batch

        attn_maps = self.model(image)

        attn_maps = attn_maps[:, 0:17, :, :]
        # loss = torch.mean(torch.pow(attn_maps - heatmap, 2) * mask[:, :, :, None])
        bs, c, h, w = attn_maps.shape
        attn_maps = attn_maps.reshape(bs, c, h*w)
        attn_maps = attn_maps / attn_maps.sum(dim=-1, keepdim=True)
        # sar loss
        heatmap = heatmap.reshape(bs, c, h*w)
        mask = mask[:, :, 0]
        loss = -torch.log((attn_maps * heatmap).sum(dim=-1) + 1e-6) * mask
        loss = loss.sum() / mask.sum()
        return loss
    
    def configure_optimizers(self):
        lr = self.lr
        params_grad = [n for n, p in self.model.named_parameters() if p.requires_grad]
        print("param_grad: {}".format(params_grad))
        # optimize context to maximize attention at pixel_loc
        parameters = [
            {
                "params": self.model.context,
                "lr": lr,
            },
            {
                "params": [p for n, p in self.model.unet.named_parameters() if p.requires_grad],
                "lr": lr * 0.1,
            },
        ]
        optimizer = torch.optim.Adam(parameters)
        scheduler = StepLR(optimizer, step_size=80, gamma=0.1)
        return [optimizer], [scheduler]
    
class GenLocSD15(nn.Module):
    def __init__(self, model_path, res=64, context_path=768, lora_path=''):
        super().__init__()
        from .sdlib import load_ldm, image2latent, collect_maps
        ldm, controller = load_ldm(model_path, res, lora_path)
        self.vae = ldm.vae
        self.unet = ldm.unet
        self.controller = controller
        if type(context_path) == int:
            context = torch.randn(1, 1000, context_path)
            self.context = nn.Parameter(context)
        elif type(context_path) == str:
            context = torch.load(context_path)
            self.context = nn.Parameter(context)

        self.image2latent = image2latent
        self.collect_maps = collect_maps

    def forward(self, image):
        with torch.no_grad():
            latent = self.image2latent(self.vae, image)
        
        context = self.context.expand(image.shape[0], -1, -1)
        noisy_image = latent
        pred_noise = self.unet(noisy_image, torch.zeros_like(context[:, 0, 0]), context)["sample"]

        attention_maps = self.collect_maps(self.controller)
        self.controller.reset()
        
        return attention_maps