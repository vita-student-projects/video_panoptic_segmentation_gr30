from bit_diffusion import Encoder, EncoderDecoder, Unet, BitDiffusion
from trainer import Trainer
from utils import seed_everything
from dataset import get_train_loaders_kittistep
from config import CFG, ImageShape
from torchinfo import summary

if __name__ == '__main__':
    seed_everything(42)

    train_loader, valid_loader = get_train_loaders_kittistep()
    print("Train size: ", len(train_loader.dataset))
    print("Valid size: ", len(valid_loader.dataset))

    model = EncoderDecoder(
        Encoder(freeze_backbone=not CFG["train"]["freeze_backbone"]),
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
    
    summary(model)

    trainer = Trainer(
        model,
        # fn to get loaders
        get_train_loaders_kittistep,
        # where to save results
        results_folder='./results/train',
        # number of samples
        num_samples=16,
        # training batch size
        train_batch_size=CFG["train"]["batch_size"],
        # gradient accumulation
        gradient_accumulate_every=CFG["train"]["accumulate_every"],
        # learning rate
        train_lr=CFG["train"]["lr"],
        # how often to save and sample
        save_and_sample_every=500,
        # total training steps
        train_num_steps=10000,
        # exponential moving average decay
        ema_decay=0.995,
    )

    trainer.resume_training()
