import math
from pathlib import Path

from utils import cycle, exists, has_int_squareroot
from checkpoint_surgery import perform_checkpoint_surgery, perform_checkpoint_surgery_ema

import torch
from torch.optim import Adam
from ema_pytorch import EMA
from torchvision import utils
from tqdm.auto import tqdm
from accelerate import Accelerator

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        loaders_fn,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='./results',
        amp=False,
        fp16=False,
        split_batches=True
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )

        self.accelerator.native_amp = amp

        self.model = diffusion_model

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.decoder.image_size

        # dataset and dataloader

        # self.ds = Dataset(folder, self.image_size,
        #   augment_horizontal_flip=augment_horizontal_flip, convert_image_to=pil_img_type)
        # dl = DataLoader(self.ds, batch_size=train_batch_size,
        #                 shuffle=True, pin_memory=True, num_workers=cpu_count())
        
        dl, val_dl = loaders_fn()

        dl = self.accelerator.prepare(dl)
        val_dl = self.accelerator.prepare(val_dl)
        self.dl = cycle(dl)
        self.val_dl = cycle(val_dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(),
                        lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'))

        ckpt = data['model']
        if milestone == 0:
            ckpt = perform_checkpoint_surgery(data['model'])
        
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(ckpt)

        self.step = data['step']
        if milestone == 0:
            self.step = 0
        
        if milestone != 0:
            self.opt.load_state_dict(data['opt'])
        
        ckpt = data['ema']
        if milestone == 0:
            ckpt = perform_checkpoint_surgery_ema(data['ema'])
        
        self.ema.load_state_dict(ckpt)

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def resume_training(self):
        import glob
        import re
        import pathlib

        checkpoints = glob.glob(str(self.results_folder / "model-*"))
        
        
        checkpoints = [
            (x, int(re.findall(r"(\d+)", x)[0]))
            for x in checkpoints
        ]
        checkpoints = sorted(checkpoints, key=lambda x: x[1])[::-1]
        
        if len(checkpoints) != 0:
            latest_checkpoint = checkpoints[0]
            
            print(f"Resuming training from checkpoint: {pathlib.Path(latest_checkpoint[0]).name}")
            self.load(milestone=latest_checkpoint[1])
        else:
            print("No checkpoints found: starting training from scratch")

        self.train()
            
    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    x, y = next(self.dl)
                    x, y = [el.to(device) for el in x], [el.to(device) for el in y]

                    with self.accelerator.autocast():
                        loss = self.model(x, y)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            x, y = next(self.val_dl)
                            x, y = [el.to(device) for el in x], [el.to(device) for el in y]

                            milestone = self.step // self.save_and_sample_every
                            pred, val_loss = self.ema.ema_model.sample(x[-1], y[:-1], ground_truth=y[-1])

                        utils.save_image(pred, str(
                            self.results_folder / f'sample-{milestone}.png'), nrow=int(math.sqrt(self.num_samples)))
                        
                        # all_images = y[-1]
                        # utils.save_image(all_images, str(
                        #     self.results_folder / f'sample-{milestone}-ground-truth.png'), nrow=int(math.sqrt(self.num_samples)))
                        self.save(milestone)
                        
                        with open(self.results_folder / 'train_log.txt', 'a') as f:
                            f.write(f"Iter: {self.step:5d} - Train Loss: {total_loss:.6f} - Val Loss: {val_loss:.6f} \n")

                self.step += 1
                pbar.update(1)

        accelerator.print('training complete')
