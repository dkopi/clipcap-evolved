import bitsandbytes as bnb
from train import  MLP, COCODataset, CosineWarmupScheduler
from peft import LoraConfig, TaskType, LoraModel, get_peft_model
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    CLIPModel,
    CLIPProcessor,
)
from typing import Tuple, Optional
import argparse
import torch.nn as nn
from clipcap_transformer import ClipCapTransformerMapper
import os
import signal
import torch
from torch.utils.data import DataLoader
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.plugins.environments import SLURMEnvironment
import wandb



class LoRACaptioningModel(nn.Module):
    def __init__(
        self,
        prefix_length: int = 8,
        mlp_hidden_size: int = 64,
        visual_output_size: int = 768,
        gpt2_pretrained_model="gpt2",
        clip_pretrained_model="openai/clip-vit-base-patch32",
        use_unpooled_output: bool = False,
        architecture: str = "mlp",
    ):
        super().__init__()

        self.prefix_length = prefix_length
        self.use_unpooled_output = use_unpooled_output
        self.architecture = architecture
        if use_unpooled_output:
            self.visual_output_size = visual_output_size * 50
        else:
            self.visual_output_size = visual_output_size

        self.clip = CLIPModel.from_pretrained(clip_pretrained_model)


        #print(lora_model.print_trainable_parameters())

        #self.gpt = lora_model
        self.gpt = GPT2LMHeadModel.from_pretrained(gpt2_pretrained_model, load_in_8bit=True)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        if architecture == "mlp":
            self.mapper = MLP(
                self.visual_output_size,
                mlp_hidden_size,
                self.gpt_embedding_size * prefix_length,
            )
        elif architecture == "clipcap":
            self.mapper = ClipCapTransformerMapper(
                self.visual_output_size, self.gpt_embedding_size, prefix_length, 10, 8
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # todo: indicate in the docs that by default these parts are frozen and put to eval mode
        self.clip.eval()
        self.gpt.eval()
        for param in self.clip.parameters():
            param.requires_grad = False
        for param in self.gpt.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(peft_type="LORA", task_type="CAUSAL_LM",
                            r=4,
                            lora_alpha=32,
                            target_modules=["c_attn"],
                            lora_dropout=0.01,
                            fan_in_fan_out=True,
                            )
        #print(self.gpt)
        self.gpt = get_peft_model(self.gpt, lora_config)
        self.gpt.print_trainable_parameters()

    def parameters(self, recurse: bool = True):
        return self.mapper.parameters()

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        with torch.no_grad():
            if self.use_unpooled_output:
                prefix = self.clip.vision_model(images)["last_hidden_state"].flatten(
                    start_dim=-2
                )
            else:
                prefix = self.clip.vision_model(images)["pooler_output"]
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.mapper(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, attention_mask=mask)
        return out


class LoRATrainingModule(pl.LightningModule):
    def __init__(
        self,
        prefix_length,
        mlp_hidden_size,
        use_unpooled_output,
        epochs,
        arch,
        samples_per_epoch,
        lr=1e-4,
        warmup=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = LoRACaptioningModel(
            prefix_length,
            mlp_hidden_size,
            use_unpooled_output=use_unpooled_output,
            architecture=arch,
        )
        self.loss_module = nn.CrossEntropyLoss(ignore_index=0)

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        return self.model(tokens, images, mask)

    def configure_optimizers(self):
        # make it more flexible
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        if self.hparams.warmup is not None:
            lr_scheduler = CosineWarmupScheduler(
                optimizer,
                # todo: make it equal for whole epochs?
                warmup=self.hparams.warmup * self.hparams.samples_per_epoch,
                max_iters=self.hparams.epochs * self.hparams.samples_per_epoch,
                # optimizer, warmup=self.hparams.warmup, max_iters=self.hparams.max_iters
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
        else:
            return optimizer

    def _gradient_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)
        return total_norm

    def training_step(self, batch, batch_idx):
        tokens, mask, image = batch
        output = self(tokens, image, mask)
        logits = output.logits[:, self.hparams.prefix_length - 1 : -1]
        loss = self.loss_module(logits.reshape(-1, logits.shape[-1]), tokens.flatten())
        self.log(
            "grad_norm", self._gradient_norm(self.model), on_step=True, on_epoch=True
        )
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # todo: put into eval mode? or is it done automatically?
        tokens, mask, image = batch
        with torch.no_grad():
            output = self(tokens, image, mask)
        logits = output.logits[:, self.hparams.prefix_length - 1 : -1]
        loss = self.loss_module(logits.reshape(-1, logits.shape[-1]), tokens.flatten())
        self.log("val_loss", loss, prog_bar=True)

def train_model(
    annotations_file,
    data_dir,
    val_annotations_file,
    val_data_dir,
    batch_size,
    run_name=None,
    save_name="default",
    checkpoint_path="checkpoints",
    find_lr=False,
    **kwargs,
):
    if int(os.environ.get("SLURM_RESTART_COUNT", 0)) > 0:
        with open(os.path.join(checkpoint_path, "last_run_id"), "r") as f:
            run_id = f.read()
    else:
        run_id = wandb.util.generate_id()
    wandb_logger = WandbLogger(
        project="clipcap_evolved",
        name=run_name,
        entity="clipcap-dl2",
        id=run_id,
        resume="allow",
        log_model=False,
    )
    os.makedirs(checkpoint_path, exist_ok=True)
    with open(os.path.join(checkpoint_path, "last_run_id"), "w") as f:
        f.write(run_id)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_set = COCODataset(
        annotations_file=annotations_file,
        data_dir=data_dir,
        prefix_length=kwargs["prefix_length"],
    )
    val_set = COCODataset(
        annotations_file=val_annotations_file,
        data_dir=val_data_dir,
        prefix_length=kwargs["prefix_length"],
        sample_limit=1000,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        # pin_memory=True,
        num_workers=16,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        drop_last=False,
        num_workers=16,
    )

    trainer = pl.Trainer(
        default_root_dir=os.path.join(checkpoint_path, save_name),
        accelerator="gpu" if not str(device).startswith("cpu") else "cpu",
        devices=1,
        max_epochs=kwargs["epochs"],
        callbacks=[
            # ModelCheckpoint(
            #     save_weights_only=True, mode="max", monitor="val_acc"
            # ),  # Save the best checkpoint based on the maximum val_acc recorded. Saves only weights and not optimizer
            LearningRateMonitor("step"),
            # LearningRateMonitor("epoch"),
        ],
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=wandb_logger,
        log_every_n_steps=10,
        val_check_interval=1000,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
    )

    # trainer.logger._log_graph = (
    #     True  # If True, we plot the computation graph in tensorboard
    # )
    trainer.logger._default_hp_metric = (
        None  # Optional logging argument that we don't need
    )

    pl.seed_everything(42)
    model = LoRATrainingModule(samples_per_epoch=len(train_loader), **kwargs)

    wandb.run.summary["total_params"] = sum(p.numel() for p in model.parameters())
    wandb.run.summary["trainable_params"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    if find_lr:
        print("using lr finder")
        tuner = Tuner(trainer)
        tuner.lr_find(model, train_loader, val_loader)

    trainer.fit(model, train_loader, val_loader)

    # model = CIFARModule.load_from_checkpoint(
    #     trainer.checkpoint_callback.best_model_path
    # )  # Load best checkpoint after training

    # Test best model on validation and test set
    # val_result = trainer.test(model, val_loader, verbose=False)
    # test_result = trainer.test(model, test_loader, verbose=False)
    # result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    wandb.finish()

    return model




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations_file", default="./data/coco/annotations/captions_train2014.json"
    )
    parser.add_argument(
        "--val_annotations_file",
        default="./data/coco/annotations/captions_val2014.json",
    )
    parser.add_argument("--data_dir", default="./data/coco/train2014")
    parser.add_argument("--val_data_dir", default="./data/coco/val2014")
    parser.add_argument("--checkpoint_path", default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--prefix_length", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mlp_hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--find_lr", action="store_true")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--use_unpooled_output", action="store_true")
    parser.add_argument("--arch", default="mlp", choices=["mlp", "clipcap"])

    args = parser.parse_args()

    print("======= args =======")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("====================")

    train_model(
        annotations_file=args.annotations_file,
        data_dir=args.data_dir,
        val_annotations_file=args.val_annotations_file,
        val_data_dir=args.val_data_dir,
        checkpoint_path=args.checkpoint_path,
        epochs=args.epochs,
        prefix_length=args.prefix_length,
        batch_size=args.batch_size,
        mlp_hidden_size=args.mlp_hidden_size,
        lr=args.lr,
        find_lr=args.find_lr,
        run_name=args.run_name,
        warmup=args.warmup,
        use_unpooled_output=args.use_unpooled_output,
        arch=args.arch,
    )


if __name__ == "__main__":
    main()
