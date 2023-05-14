import os
import signal
from typing import Tuple, Optional
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import lightning as pl
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    TQDMProgressBar,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.plugins.environments import SLURMEnvironment
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    CLIPModel,
    CLIPProcessor,
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from PIL import Image
from pycocotools.coco import COCO
import wandb
import numpy as np
from clipcap_transformer import ClipCapTransformerMapper
from evaluate import generate, evaluate
from peft import LoraConfig, get_peft_model


class COCODataset(Dataset):
    def __init__(
        self,
        annotations_file,
        data_dir,
        clip_processor,
        prefix_length: int,
        tokenizer: None,
        max_seq_len: int = 36,
        sample_limit: Optional[int] = None,
    ):
        self.coco = COCO(annotations_file)
        self.data_dir = data_dir
        self.prefix_length = prefix_length
        self.max_seq_len = max_seq_len
        self.sample_limit = sample_limit
        print(f"dataset size: {len(self.coco.imgs)}")

        self.tokenizer = tokenizer
        self.clip_processor = clip_processor

    def __len__(self):
        return (
            len(self.coco.imgs)
            if self.sample_limit is None
            else (
                self.sample_limit
                if self.sample_limit < len(self.coco.imgs)
                else len(self.coco.imgs)
            )
        )

    def pad_tokens(self, caption: str):
        tokens = torch.tensor(self.tokenizer.encode(caption), dtype=torch.int64)
        padding = self.max_seq_len - tokens.shape[0]
        if padding > 0:
            tokens = torch.cat((tokens, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            tokens = tokens[: self.max_seq_len]
        mask = tokens.ge(0)
        tokens[~mask] = 0
        mask = mask.float()
        mask = torch.cat((torch.ones(self.prefix_length), mask), dim=0)
        return tokens, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        img_id = list(self.coco.imgs.keys())[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = f"{self.data_dir}/{img_info['file_name']}"
        raw_image = Image.open(img_path).convert("RGB")
        image = self.clip_processor(images=raw_image, return_tensors="pt").pixel_values
        image = image.squeeze(0)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        captions = [ann["caption"] for ann in anns]
        caption = captions[0]
        tokens, mask = self.pad_tokens(caption)
        return tokens, mask, image


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # todo: try different activations
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CaptioningModel(nn.Module):
    def __init__(
        self,
        direct,
        prefix_length: int = 8,
        mlp_hidden_size: int = 64,
        visual_output_size: int = 768,
        gpt2_pretrained_model="gpt2",
        flan_pretrained_model="google/flan-t5-base",
        clip_pretrained_model="openai/clip-vit-base-patch32",
        use_unpooled_output: bool = False,
        architecture: str = "mlp",
        mlp_dropout: float = 0.2,
    ):
        super().__init__()

        self.prefix_length = prefix_length
        self.use_unpooled_output = use_unpooled_output
        self.architecture = architecture
        if use_unpooled_output:
            self.visual_output_size = visual_output_size * 50
        else:
            self.visual_output_size = visual_output_size
        self.direct = direct
        self.clip = CLIPModel.from_pretrained(clip_pretrained_model)

        if architecture == "mlp":  # split, based on head and used mapper
            self.lm = GPT2LMHeadModel.from_pretrained(gpt2_pretrained_model)
            self.lm_embedding_size = self.lm.transformer.wte.weight.shape[1]
            if not self.direct:
                self.mapper = MLP(
                    self.visual_output_size,
                    mlp_hidden_size,
                    self.lm_embedding_size * prefix_length,
                    dropout=mlp_dropout,
                )
        elif architecture == "clipcap":
            self.lm = GPT2LMHeadModel.from_pretrained(gpt2_pretrained_model)
            self.lm_embedding_size = self.lm.transformer.wte.weight.shape[1]
            if not self.direct:
                self.mapper = ClipCapTransformerMapper(
                    self.visual_output_size,
                    self.lm_embedding_size,
                    prefix_length,
                    10,
                    8,
                )
        elif architecture == "flan-t5":
            self.lm = T5ForConditionalGeneration.from_pretrained(flan_pretrained_model)
            self.lm_embedding_size = self.lm.get_input_embeddings().weight.shape[1]
            if not self.direct:
                self.mapper = MLP(
                    self.visual_output_size,
                    mlp_hidden_size,
                    self.lm_embedding_size * prefix_length,
                )

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def token_to_embed(self, tokens: torch.Tensor):
        if self.architecture == "flan-t5":
            return self.lm.get_input_embeddings()(tokens)
        else:
            return self.lm.transformer.wte(tokens)

    def get_logits(
        self, image_embeds: torch.Tensor, encoder_outputs: Optional[torch.tensor] = None
    ):
        if self.architecture == "flan-t5":
            # Get the logits of the next token when using flan-t5
            return self.lm(
                inputs_embeds=image_embeds, decoder_inputs_embeds=encoder_outputs
            ).logits
        else:
            return self.lm(inputs_embeds=image_embeds).logits

    def get_image_embeds(self, images: torch.Tensor):
        if self.direct:
            clip_embeds = self.clip.vision_model(images)["last_hidden_state"]
            return clip_embeds
        elif self.use_unpooled_output:
            clip_embeds = self.clip.vision_model(images)["last_hidden_state"].flatten(
                start_dim=-2
            )
        else:
            clip_embeds = self.clip.vision_model(images)["pooler_output"]
        return self.mapper(clip_embeds).view(
            -1, self.prefix_length, self.lm_embedding_size
        )

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        image_embeds = self.get_image_embeds(images)

        if self.architecture == "flan-t5":
            out = self.lm(inputs_embeds=image_embeds, labels=tokens)
        elif (
            self.architecture == "clipcap" or self.architecture == "mlp"
        ):  # GPT for now
            embedding_text = self.lm.transformer.wte(tokens)
            embedding_cat = torch.cat((image_embeds, embedding_text), dim=1)
            out = self.lm(inputs_embeds=embedding_cat, attention_mask=mask)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        return out


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class TrainingModule(pl.LightningModule):
    def __init__(
        self, prefix_length, mlp_hidden_size, use_unpooled_output, arch, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = CaptioningModel(
            kwargs["direct"],
            prefix_length,
            mlp_hidden_size,
            use_unpooled_output=use_unpooled_output,
            architecture=arch,
            mlp_dropout=kwargs["mlp_dropout"],
        )
        self.freeze_target(self.model.clip)
        if not kwargs["finetune_lm"]:
            self.freeze_target(self.model.lm)
            if kwargs["lora"]:
                self.model.lm = self.get_lora_model(self.model.lm, arch)

        self.loss_module = nn.CrossEntropyLoss(ignore_index=0)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.total_params = sum(p.numel() for p in self.model.parameters())

    def freeze_target(self, target):
        for param in target.parameters():
            param.requires_grad = False

        target.eval()
    def get_lora_model(self, model, arch, lora_config = None):

        if not lora_config:
            if arch == "flan-t5":
                target_modules = ["q", "v"]
            elif arch == "clipcap":
                target_modules = ["c_attn"]
            elif arch == "mlp":
                target_modules = ["c_attn"]
            else:
                raise ValueError(f"Unknown architecture: {arch}")

            lora_config = LoraConfig(peft_type="LORA", task_type="CAUSAL_LM",
                                     r=4,
                                     lora_alpha=32,
                                     target_modules=target_modules,
                                     lora_dropout=0.01,
                                     fan_in_fan_out=True,
                                     )

        lora_model = get_peft_model(model, lora_config)
        print("h1")
        lora_model.print_trainable_parameters()
        return lora_model


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

    def get_loss(self, tokens, images, mask):
        output = self(tokens, images, mask)

        if self.hparams.arch == "flan-t5":
            loss = self.loss_module(
                output.logits.reshape(-1, output.logits.shape[-1]), tokens.flatten()
            )  # TODO: Do we use our loss or loss from the T5?
        elif self.hparams.arch == "mlp" or self.hparams.arch == "clipcap":
            logits = output.logits[:, self.hparams.prefix_length - 1 : -1]
            loss = self.loss_module(
                logits.reshape(-1, logits.shape[-1]), tokens.flatten()
            )
        return loss

    def training_step(self, batch, batch_idx):
        tokens, mask, images = batch
        loss = self.get_loss(tokens, images, mask)

        self.log(
            "grad_norm", self._gradient_norm(self.model), on_step=True, on_epoch=False
        )
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        # todo: put into eval mode? or is it done automatically?
        with torch.no_grad():
            tokens, mask, images = batch

            if batch_idx == 0:
                for i in range(5):
                    to_caption = images[i : i + 1]
                    image_embeds = self.model.get_image_embeds(to_caption)
                    caption = generate(
                        self.model,
                        self.hparams.tokenizer,
                        image_embeds,
                        arch=self.hparams.arch,
                    )
                    print(f"\ncaption: {caption}\n")

            if batch_idx < self.hparams.eval_batches:
                scores = evaluate(
                    self.model,
                    self.hparams.tokenizer,
                    images,
                    tokens,
                    arch=self.hparams.arch,
                )
                self.log("cider", scores["cider"], prog_bar=True)

            loss = self.get_loss(tokens, images, mask)

            self.log("val_loss", loss, prog_bar=True)
            self.log("trainable_params", float(self.trainable_params))
            self.log("total_params", float(self.total_params))


def train_model(
    batch_size,
    run_name=None,
    save_name="default",
    checkpoint_path="checkpoints",
    find_lr=False,
    **kwargs,
):
    pl.seed_everything(42)

    data_dir = kwargs["data_dir"]
    val_data_dir = kwargs["val_data_dir"]
    annotations_file = kwargs["annotations_file"]
    val_annotations_file = kwargs["val_annotations_file"]

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
        offline=kwargs["offline"],
    )
    wandb_logger.experiment.config["job_id"] = os.environ.get("SLURM_JOB_ID", None)

    os.makedirs(checkpoint_path, exist_ok=True)
    with open(os.path.join(checkpoint_path, "last_run_id"), "w") as f:
        f.write(run_id)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if kwargs["arch"] == "mlp" or kwargs["arch"] == "clipcap":  # lm_model
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif kwargs["arch"] == "flan-t5":
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    train_set = COCODataset(
        annotations_file=annotations_file,
        data_dir=data_dir,
        prefix_length=kwargs["prefix_length"],
        tokenizer=tokenizer,
        clip_processor=clip_processor,
    )
    val_set = COCODataset(
        annotations_file=val_annotations_file,
        data_dir=val_data_dir,
        prefix_length=kwargs["prefix_length"],
        sample_limit=1000,
        clip_processor=clip_processor,
        tokenizer=tokenizer,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        # pin_memory=True,
        # num_workers=16,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        drop_last=False,
        # num_workers=16,
    )

    plugins = []
    if os.environ.get("SLURM_JOB_ID"):
        plugins.append(SLURMEnvironment(requeue_signal=signal.SIGUSR1))

    training_module = TrainingModule(
        samples_per_epoch=len(train_loader),
        tokenizer=tokenizer,
        clip_processor=clip_processor,
        **kwargs,
    )
    wandb_logger.experiment.config[
        "trainable_params"
    ] = training_module.trainable_params
    wandb_logger.experiment.config["total_params"] = training_module.total_params

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
            TQDMProgressBar(refresh_rate=10),
        ],
        enable_checkpointing=False,
        enable_progress_bar=True,
        logger=wandb_logger,
        log_every_n_steps=10,
        val_check_interval=kwargs["val_freq"],
        plugins=plugins,
        gradient_clip_val=kwargs["grad_clip"],
    )

    trainer.logger._default_hp_metric = None

    if find_lr:
        print("using lr finder")
        tuner = Tuner(trainer)
        tuner.lr_find(training_module, train_loader, val_loader)

    trainer.fit(training_module, train_loader, val_loader)

    # model = CIFARModule.load_from_checkpoint(
    #     trainer.checkpoint_callback.best_model_path
    # )  # Load best checkpoint after training

    # Test best model on validation and test set
    # val_result = trainer.test(model, val_loader, verbose=False)
    # test_result = trainer.test(model, test_loader, verbose=False)
    # result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    wandb.finish()

    return training_module


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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--prefix_length", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mlp_hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--find_lr", action="store_true")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--use_unpooled_output", action="store_true")
    parser.add_argument("--arch", default="mlp", choices=["mlp", "clipcap", "flan-t5"])
    parser.add_argument("--eval_batches", type=int, default=16)
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--finetune_lm", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--val_freq", type=int, default=1000)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--direct", action="store_true")

    args = parser.parse_args()

    if args.direct:
        args.prefix_length = 50
        args.finetune_lm = True
    if args.lora:
        args.finetune_lm = True

    print("======= args =======")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("====================")

    train_model(**vars(args))


if __name__ == "__main__":
    main()
