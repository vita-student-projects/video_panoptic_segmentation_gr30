from bit_diffusion import Encoder, EncoderDecoder, Unet, BitDiffusion
from trainer import Trainer
from utils import seed_everything
from dataset import get_train_loaders_kittistep
from config import CFG, ImageShape
from collections import deque
import glob
import pathlib
from PIL import Image
from torchvision import transforms
from argparse import ArgumentParser
import torch
import re
from tqdm.auto import tqdm
from torchvision import transforms

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./results/train/model-latest.pt")
    parser.add_argument("--input_folder", type=str, default="./data/input")
    parser.add_argument("--output_folder", type=str, default="./data/output")
    parser.add_argument("--use_frames_before", type=bool, default=False)
    
    args = parser.parse_args()
    
    model = EncoderDecoder(
        Encoder(),
        BitDiffusion(
            Unet(
                dim=32,
                channels=3,
                embedding_channels=32,
                dim_mults=(1, 2, 4, 8),
            ),
            image_size=CFG["image_size"],
            embedding_size=ImageShape(width=304, height=88),
            timesteps=100,
            time_difference=0.1,
            use_ddim=True
        )
    )
    
    model = model.to(CFG["device"])
    
    checkpoint = torch.load(args.checkpoint)["model"]
    model.load_state_dict(checkpoint)
    
    
    n_frames = CFG["misc"]["frames_before"]
    condition_q = deque([torch.zeros((1, 3, 88, 304), device=CFG["device"]) for _ in range(n_frames)], maxlen=n_frames)
    condition_q = deque([torch.zeros((1, 3, CFG["image_size"].height, CFG["image_size"].width), device=CFG["device"]) for _ in range(n_frames)], maxlen=n_frames)
    
    input_path = pathlib.Path(args.input_folder)
    images = glob.glob(str(input_path / "*left*"))
    images = [
        (x, int(re.findall(r"(\d+)", x)[1]))
        for x in images
    ]
    images = sorted(images, key=lambda x: x[1])
    
    output_path = pathlib.Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)    
    
    with torch.no_grad():
        with tqdm(initial=1, total=len(images)+1) as pbar:
            
            center_crop = transforms.CenterCrop((CFG["image_size"].height, CFG["image_size"].width))
            to_tensor = transforms.ToTensor()
            to_pil = transforms.ToPILImage()
            
            for image, index in images:
                if args.use_frames_before and index < n_frames:
                    pred = Image.open(str(output_path / f"pred-{index:06d}.png"))
                    pred = pred.resize((CFG["image_size"].width, CFG["image_size"].height))
                    condition_q.append(to_tensor(pred).to(CFG["device"]).unsqueeze(0))
                    print(f"Skipping generation of mask for frame {index:2d}, using the provided one")
                    continue
                
                image = Image.open(image)
                cropped_image = center_crop(image)
                 
                x = to_tensor(cropped_image).to(CFG["device"]).unsqueeze(0)
                pred = model.sample(x, list(condition_q))
                
                pred_img = to_pil(pred[0]).resize(cropped_image.size)
                pred_img.save(str(output_path / f"pred-{index:06d}.png"))
                
                condition_q.append(to_tensor(pred_img).to(CFG["device"]).unsqueeze(0))
                
                pbar.update(1)
            
    