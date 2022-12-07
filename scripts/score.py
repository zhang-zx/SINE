import clip
import torch
import os
from PIL import Image
import lpips
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from einops import repeat
import numpy as np
import pickle
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
    
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str, required=True)
parser.add_argument('--mode', type=str, default='K', choices=['K', 'beta'])
parser.add_argument('--prompt', type=str, required=True)
parser.add_argument('--orig_img', type=str, required=True)
opt = parser.parse_args()


def _transform():
    return Compose([
        Resize((512, 512), interpolation=BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
def _convert_image_to_rgb(image):
    return image.convert("RGB")

ORIG_IMAGE_PATH = opt.orig_img
TGT_TEXT = opt.prompt
IMAGE_DIR = opt.img_dir

# import ipdb
# ipdb.set_trace()
if opt.mode == 'K':
    Xs = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
else:
    Xs = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
NUM_IMGS_PER_EXP = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

img_paths = [os.path.join(IMAGE_DIR, f) for f in sorted(os.listdir(IMAGE_DIR)) if os.path.isfile(os.path.join(IMAGE_DIR, f))]

loss_fn = lpips.LPIPS(net='alex').to(device)
orig_img = Image.open(ORIG_IMAGE_PATH)
transform = _transform()
orig_img = transform(orig_img).unsqueeze(0).to(device)
# orig_img = repeat(orig_img, '1 c h w -> n c h w', n=len(img_paths))
imgs = [transform(Image.open(img_path)).unsqueeze(0).to(device) for img_path in img_paths]
lpips_scores = [loss_fn(img, orig_img).item() for img in imgs]
print(lpips_scores)

model, preprocess = clip.load("ViT-B/32", device=device)


imgs = [preprocess(Image.open(img_path)).unsqueeze(0).to(device) for img_path in img_paths]
text_tokens = clip.tokenize(TGT_TEXT).to(device)
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    img_features = model.encode_image(torch.cat(imgs, dim=0))
    img_features /= img_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * img_features @ text_features.T).squeeze()

print(similarity)
l_score_exps = list()
c_score_exps = list()
x_axis = list()
for i in range(len(img_paths)):
    if i % NUM_IMGS_PER_EXP == 0:
        l_score_exps.append(list())
        c_score_exps.append(list())
        x_axis.append([Xs[i//NUM_IMGS_PER_EXP]] * NUM_IMGS_PER_EXP)
    l_score_exps[-1].append(lpips_scores[i])
    c_score_exps[-1].append(similarity[i].item())
    
result = dict()
result['l_score_exps'] = l_score_exps
result['c_score_exps'] = c_score_exps
result['x_axis'] = x_axis

pickle.dump(result, open(os.path.join(os.path.dirname(IMAGE_DIR), 'scores.pkl'), 'wb'))

# import ipdb
# ipdb.set_trace()
# print([np.mean(l_score) for l_score in l_score_exps])
# print([np.mean(c_score) for c_score in c_score_exps])

    


