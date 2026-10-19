import os
#import wandb
import numpy as np
import torch
import numpy as np
from tqdm import tqdm
import json
import time
import torch.nn.functional as F
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval
import random
import shutil
import lightning as L

from peft.utils import get_peft_model_state_dict
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers import StableDiffusionPipeline

from dataset.transforms import transform_preds
from dataset.ap10k import AP10KDataset
from model.genloc import GenLocSD15, GenLocPLModel

dataset_name = 'ap10k'

# optimization
model_path = "/path/stable-diffusion-v1-5"
condition_dim = 768
ckpt_path = "runs/" + dataset_name + "/epoch90"
lr = 1e-2
batch_size = 2
output_heatmap_res = 64
trainer = L.Trainer(
    max_epochs=90,
    accumulate_grad_batches=50,
    devices=2,
    accelerator="gpu",
    strategy='ddp_find_unused_parameters_true',
    precision="16-mixed",
    default_root_dir=ckpt_path,
    enable_checkpointing=False
)

# if ckpt_path doesnt exist create it
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
    
# print number of gpus
print("Number of GPUs: ", torch.cuda.device_count())

# save code
src_folder = os.path.join(ckpt_path, 'src')
if not os.path.exists(src_folder):
    os.mkdir(src_folder)
    shutil.copytree('dataset', os.path.join(src_folder, 'datasets'))
    shutil.copytree('model', os.path.join(src_folder, 'models'))
    shutil.copy2('train.py', src_folder)
else:
    print("=> src files are already existed in: {}".format(src_folder))
# load data
dataset = AP10KDataset(root='data/ap-10k/', image_set='annotations/ap10k-train-split1.json', is_train=True, output_res=output_heatmap_res)
num_joints = 17

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)

model = GenLocSD15(model_path, output_heatmap_res, context_path=condition_dim)
pl_model = GenLocPLModel(model, learning_rate=lr)
start = time.time()

trainer.fit(model=pl_model, train_dataloaders=dataloader)

print(f"optimization took {time.time() - start} seconds")

embedding = pl_model.model.context.data.detach()
torch.save(embedding, os.path.join(ckpt_path, "embedding.pt"))
torch.cuda.empty_cache()

# save lora params
unet_lora_state_dict = convert_state_dict_to_diffusers(
    get_peft_model_state_dict(pl_model.model.unet)
)
StableDiffusionPipeline.save_lora_weights(
    save_directory=os.path.join(ckpt_path, "lora"),
    unet_lora_layers=unet_lora_state_dict,
    safe_serialization=True,
)

# evaluate
model = GenLocSD15(model_path, output_heatmap_res, context_path=os.path.join(ckpt_path, 'embedding.pt'), lora_path=os.path.join(ckpt_path, "lora"))
model = model.cuda()

all_preds = []
dataset = AP10KDataset(root='data/ap-10k/', image_set='annotations/ap10k-val-split1.json', is_train=False, output_res=output_heatmap_res)
num_joints = 17

# create dataloader for the dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=False, drop_last=False)
dataloader_iter = iter(dataloader)

for i in tqdm(range(len(dataloader))):

    batch = next(dataloader_iter)

    image, heatmap, keypoint, mask, meta = batch
    image_flip = image.flip(3)
    image = torch.cat((image, image_flip), dim=0)
    with torch.no_grad():
        attn_maps = model(image.cuda())
    attention_maps = attn_maps[:, :num_joints, :, :]
    bs, c, h, w = attention_maps.shape
    attn_maps = attention_maps.reshape(bs, c, h*w)
    attention_maps = attention_maps / attn_maps.sum(dim=-1, keepdim=True)[:, :, :, None]

    attention_maps, attention_maps_flip = torch.chunk(attention_maps, 2, dim=0)
    flip_index = [1, 0, 2, 3, 4, 8, 9, 10, 5, 6, 7, 14, 15, 16, 11, 12, 13]
    attention_maps_flip = attention_maps_flip[:, flip_index, :, :]
    attention_maps_flip = attention_maps_flip.flip(3)
    attention_maps_flip[:, :, :, 1:] = attention_maps_flip[:, :, :, :-1]
    attention_maps = (attention_maps + attention_maps_flip) / 2.0

    for idx in range(len(attention_maps)):
        batch_size, h, w = attention_maps[idx].shape
        map_reshaped = attention_maps[idx].view(batch_size, -1)
        maxvals, maxinds = map_reshaped.max(dim=-1)
        xs, ys = maxinds % w, (maxinds / w).long().float()
        pred_kpts = torch.stack((xs, ys), dim=1)
        pred_kpts = pred_kpts.cpu().numpy()
        for k in range(batch_size):
            heatmap = attention_maps[idx][k].cpu().numpy()
            px = int(pred_kpts[k][0])
            py = int(pred_kpts[k][1])
            if 1 < px < w - 1 and 1 < py < h - 1:
                diff = np.array([
                    heatmap[py][px + 1] - heatmap[py][px - 1],
                    heatmap[py + 1][px] - heatmap[py - 1][px]
                ])
                pred_kpts[k] += np.sign(diff) * .25
                
        c = meta['center'].numpy()
        s = meta['scale'].numpy()
        score = meta['score'].numpy()
        kpts = transform_preds(
            pred_kpts, c[idx], s[idx], [w, h]
        )
        kpts = np.concatenate((kpts, maxvals[:, None].cpu().numpy()), axis=1)

        kpts = kpts.reshape(-1).tolist()
        data = dict()
        data['image_id'] = int(meta['img_id'][idx])
        data['score'] = float(np.mean(maxvals.cpu().numpy()))
        data['keypoints'] = kpts
        data['category_id'] = int(meta['cat_id'][idx])
        all_preds.append(data)

ann_file = 'data/ap-10k/annotations/ap10k-val-split1.json'
res_file = os.path.join(ckpt_path, 'pred_val_kpt.json')
with open(res_file, 'w') as fid:
    json.dump(all_preds, fid)

cocoGt = COCO(ann_file)
cocoDt = cocoGt.loadRes(res_file)

cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints', dataset.sigmas)
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()

stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)',
                'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']
info_str = {}
for ind, name in enumerate(stats_names):
    info_str[name] = cocoEval.stats[ind]

print(info_str)