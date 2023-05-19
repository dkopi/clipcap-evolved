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
from decoder_with_head import DecoderWithHead
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
    def __init__(
        self, input_size, hidden_size, output_size, dropout=0.2, activation="relu"
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "leaky":
            self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class CaptioningModel(nn.Module):
    def __init__(
        self,
        direct,
        direct_proj,
        prefix_length: int = 8,
        mlp_hidden_size: int = 64,
        visual_output_size: int = 768,
        gpt2_pretrained_model="gpt2",
        flan_pretrained_model="google/flan-t5-base",
        clip_pretrained_model="openai/clip-vit-base-patch32",
        use_unpooled_output: bool = False,
        architecture: str = "mlp",
        mlp_dropout: float = 0.2,
        activation: str = "relu",
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
        self.direct_proj = direct_proj
        self.clip = CLIPModel.from_pretrained(clip_pretrained_model).vision_model

        if architecture == "mlp" or architecture == "clipcap":
            self.lm = GPT2LMHeadModel.from_pretrained(gpt2_pretrained_model)
            self.lm_embedding_size = self.lm.transformer.wte.weight.shape[1]
        elif architecture == "flan-t5":
            self.lm = T5ForConditionalGeneration.from_pretrained(flan_pretrained_model)
            self.lm_embedding_size = self.lm.get_input_embeddings().weight.shape[1]
        elif architecture == "flan-mlp" or architecture == "flan-transformer":
            # TODO: Load only the decoder weights
            self.lm = T5ForConditionalGeneration.from_pretrained(flan_pretrained_model)
            self.lm_embedding_size = self.lm.get_input_embeddings().weight.shape[1]
            self.lm = DecoderWithHead(
                self.lm.shared, self.lm.decoder, self.lm.lm_head, self.lm.config
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        if direct_proj:
            lm_input_size = self.lm_embedding_size
        else:
            lm_input_size = self.lm_embedding_size * prefix_length

        if not self.direct:
            if architecture == "flan-transformer" or architecture == "clipcap":
                self.mapper = ClipCapTransformerMapper(
                    self.visual_output_size,
                    self.lm_embedding_size,
                    prefix_length,
                    10,
                    8,
                )
            else:
                self.mapper = MLP(
                    self.visual_output_size,
                    mlp_hidden_size,
                    lm_input_size,
                    dropout=mlp_dropout,
                    activation=activation,
                )

    def token_to_embed(self, tokens: torch.Tensor):
        if (
            self.architecture == "flan-t5"
            or self.architecture == "flan-mlp"
            or self.architecture == "flan-transformer"
        ):
            return self.lm.get_input_embeddings()(tokens)
        else:
            return self.lm.transformer.wte(tokens)

    def get_logits(
        self, image_embeds: torch.Tensor, encoder_outputs: Optional[torch.tensor] = None
    ):
        if self.architecture == "flan-t5":
            # Get the logits of the next token when using flan-t5
            # eos = torch.tensor([self.lm.config.eos_token_id]).to(image_embeds.device)
            # eos_embeds = (
            #     self.token_to_embed(eos)
            #     .unsqueeze(0)
            #     .expand(
            #         image_embeds.shape[0],
            #         1,
            #         image_embeds.shape[2],
            #     )
            # )
            # image_embeds = torch.cat((image_embeds, eos_embeds), dim=1)
            return self.lm(
                inputs_embeds=image_embeds, decoder_inputs_embeds=encoder_outputs
            ).logits
        elif self.architecture == "flan-mlp" or self.architecture == "flan-transformer":
            # eos = torch.tensor([self.lm.config.eos_token_id]).to(image_embeds.device)
            # eos_embeds = (
            #     self.token_to_embed(eos)
            #     .unsqueeze(0)
            #     .expand(
            #         image_embeds.shape[0],
            #         1,
            #         image_embeds.shape[2],
            #     )
            # )
            # image_embeds = torch.cat((image_embeds, eos_embeds), dim=1)
            return self.lm(
                hidden_states=image_embeds, decoder_inputs_embeds=encoder_outputs
            ).logits
        else:
            return self.lm(inputs_embeds=image_embeds).logits

    def get_image_embeds(self, images: torch.Tensor):
        if self.direct or self.direct_proj:
            clip_embeds = self.clip(images)["last_hidden_state"]
            if self.direct_proj:
                clip_embeds = self.mapper(clip_embeds)
            return clip_embeds
        elif self.use_unpooled_output:
            clip_embeds = self.clip(images)["last_hidden_state"].flatten(start_dim=-2)
        else:
            clip_embeds = self.clip(images)["pooler_output"]
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
            # eos = torch.tensor([self.lm.config.eos_token_id]).to(tokens.device)
            # eos_embeds = (
            #     self.token_to_embed(eos)
            #     .unsqueeze(0)
            #     .expand(
            #         image_embeds.shape[0],
            #         1,
            #         image_embeds.shape[2],
            #     )
            # )
            # image_embeds = torch.cat((image_embeds, eos_embeds), dim=1)
            out = self.lm(inputs_embeds=image_embeds, labels=tokens)
        elif self.architecture == "clipcap" or self.architecture == "mlp":
            embedding_text = self.lm.transformer.wte(tokens)
            embedding_cat = torch.cat((image_embeds, embedding_text), dim=1)
            out = self.lm(inputs_embeds=embedding_cat, attention_mask=mask)
        elif self.architecture == "flan-mlp" or self.architecture == "flan-transformer":
            # eos = torch.tensor([self.lm.config.eos_token_id]).to(tokens.device)
            # eos_embeds = (
            #     self.token_to_embed(eos)
            #     .unsqueeze(0)
            #     .expand(
            #         image_embeds.shape[0],
            #         1,
            #         image_embeds.shape[2],
            #     )
            # )
            # image_embeds = torch.cat((image_embeds, eos_embeds), dim=1)
            out = self.lm(hidden_states=image_embeds, labels=tokens)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        return out


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters, no_cosine=False):
        self.warmup = warmup
        self.no_cosine = no_cosine
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(step=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, step):
        if self.no_cosine:
            lr_factor = 1.0
        else:
            lr_factor = 0.5 * (1 + np.cos(np.pi * step / self.max_num_iters))
        if step <= self.warmup:
            lr_factor *= step * 1.0 / self.warmup
        return lr_factor


class TrainingModule(pl.LightningModule):
    def __init__(
        self, prefix_length, mlp_hidden_size, use_unpooled_output, arch, **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = CaptioningModel(
            kwargs["direct"],
            kwargs["direct_proj"],
            prefix_length,
            mlp_hidden_size,
            gpt2_pretrained_model="gpt2" + kwargs["gpt_size"],
            flan_pretrained_model="google/flan-t5-" + kwargs["flan_size"],
            use_unpooled_output=use_unpooled_output,
            architecture=arch,
            mlp_dropout=kwargs["mlp_dropout"],
        )
        self.test_dataset = "none"

        self.freeze_model()

        self.loss_module = nn.CrossEntropyLoss(ignore_index=0)
        self.clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32"
        )
        self.trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.total_params = sum(p.numel() for p in self.model.parameters())

    def freeze_model(self, skip_grad=False):
        self.freeze_target(self.model.clip, skip_grad)
        if not self.hparams.finetune_lm:
            self.freeze_target(self.model.lm, skip_grad)
            if self.hparams.lora and not skip_grad:
                if self.hparams.arch == "flan-t5":
                    self.freeze_target(self.model.lm)
                self.model.lm = self.get_lora_model(self.model.lm, self.hparams.arch)
        elif self.hparams.arch == "flan-t5":
            self.freeze_target(self.model.lm.decoder, skip_grad)

    def freeze_target(self, target, skip_grad=False):
        if not skip_grad:
            for param in target.parameters():
                param.requires_grad = False

        target.eval()

    def get_lora_model(self, model, arch, lora_config=None):
        if not lora_config:
            if arch == "flan-t5" or arch == "flan-mlp" or arch == "flan-transformer":
                target_modules = ["q", "v"]
            elif arch == "clipcap" or arch == "mlp":
                target_modules = ["c_attn", "c_proj", "c_fc", "lm_head"]
            else:
                raise ValueError(f"Unknown architecture: {arch}")

            lora_config = LoraConfig(
                peft_type="LORA",
                task_type="CAUSAL_LM",
                r=4,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.01,
                fan_in_fan_out=True,
            )

        lora_model = get_peft_model(model, lora_config)
        lora_model.print_trainable_parameters()
        return lora_model

    def forward(
        self,
        tokens: torch.Tensor,
        images: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        self.freeze_model(True)
        return self.model(tokens, images, mask)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.hparams.lr)
        if self.hparams.warmup is not None:
            lr_scheduler = CosineWarmupScheduler(
                optimizer,
                warmup=self.hparams.warmup
                if self.hparams.warmup_use_steps
                else self.hparams.warmup * self.hparams.samples_per_epoch,
                max_iters=self.hparams.epochs * self.hparams.samples_per_epoch,
                no_cosine=self.hparams.no_cosine,
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

        if (
            self.hparams.arch == "flan-t5"
            or self.hparams.arch == "flan-mlp"
            or self.hparams.arch == "flan-transformer"
        ):
            loss = self.loss_module(
                output.logits.reshape(-1, output.logits.shape[-1]), tokens.flatten()
            )
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
        with torch.no_grad():
            tokens, mask, images = batch

            if batch_idx == 0:
                image_embeds = self.model.get_image_embeds(images)
                captions = generate(
                    self.model,
                    self.hparams.tokenizer,
                    image_embeds,
                    arch=self.hparams.arch,
                )
                for caption in captions:
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

    def test_step(self, batch, batch_idx):
        tokens, mask, images = batch
        scores = evaluate(
            self.model,
            self.hparams.tokenizer,
            images,
            tokens,
            arch=self.hparams.arch,
        )
        self.log(f"{self.test_dataset}_cider", scores["cider"])


def train_model(
    batch_size,
    run_name=None,
    find_lr=False,
    **kwargs,
):
    pl.seed_everything(42)

    data_dir = kwargs["data_dir"]
    val_data_dir = kwargs["val_data_dir"]
    nocaps_dir_indomain = kwargs["nocaps_dir_indomain"]
    nocaps_dir_neardomain = kwargs["nocaps_dir_neardomain"]
    nocaps_dir_outdomain = kwargs["nocaps_dir_outdomain"]
    nocaps_annotations_file = kwargs["nocaps_annotations_file"]
    annotations_file = kwargs["annotations_file"]
    val_annotations_file = kwargs["val_annotations_file"]

    run_id = wandb.util.generate_id()

    wandb_logger = WandbLogger(
        project="clipcap_evolved",
        name=run_name,
        id=run_id,
        resume="allow",
        log_model=False,
        offline=kwargs["offline"],
    )
    wandb_logger.experiment.config["job_id"] = os.environ.get("SLURM_JOB_ID", None)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    kwargs["gpt_size"] = "-" + kwargs["gpt_size"] if kwargs["gpt_size"] != "" else ""
    if kwargs["arch"] == "mlp" or kwargs["arch"] == "clipcap":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2" + kwargs["gpt_size"])
    elif (
        kwargs["arch"] == "flan-t5"
        or kwargs["arch"] == "flan-mlp"
        or kwargs["arch"] == "flan-transformer"
    ):
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-" + kwargs["flan_size"])

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
        clip_processor=clip_processor,
        tokenizer=tokenizer,
    )
    train_set = COCODataset(
        annotations_file="./data/nocaps/annotations/near-domain.json",
        data_dir="./data/nocaps/near-domain",
        prefix_length=kwargs["prefix_length"],
        clip_processor=clip_processor,
        tokenizer=tokenizer
    )
    val_set = COCODataset(
        annotations_file="./data/nocaps/annotations/out-domain.json",
        data_dir="./data/nocaps/out-domain",
        prefix_length=kwargs["prefix_length"],
        clip_processor=clip_processor,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=16,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        drop_last=False,
        num_workers=16,
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
        default_root_dir=kwargs["checkpoint_path"],
        accelerator="gpu" if not str(device).startswith("cpu") else "cpu",
        devices=1,
        max_epochs=kwargs["epochs"],
        callbacks=[
            ModelCheckpoint(
                dirpath=kwargs["checkpoint_path"],
                save_weights_only=True,
                mode="min",
                monitor="val_loss",
            ),
            LearningRateMonitor("step"),
            TQDMProgressBar(refresh_rate=100),
        ],
        enable_progress_bar=True,
        logger=wandb_logger,
        log_every_n_steps=100,
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
    print(f"best model: {trainer.checkpoint_callback.best_model_path}")

    model = TrainingModule.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
    )

    print("final evaluation...")


    def test_model(model, dataset_name, annotations_file_path, dir_path):
        dataset = COCODataset(
            annotations_file=annotations_file_path,
            data_dir=dir_path,
            prefix_length=kwargs["prefix_length"],
            clip_processor=clip_processor,
            tokenizer=tokenizer
        )
        loader = DataLoader(
            dataset,
            batch_size=128,
            drop_last=False,
            num_workers=16,
        )
        model.test_dataset = dataset_name
        trainer.test(model, loader)

    test_model(model, "coco", val_annotations_file, val_data_dir)
    test_model(model, "nocaps_near", os.path.join(kwargs["nocaps_root"], "annotations/near-domain.json"), os.path.join(kwargs["nocaps_root"], "near-domain"))
    test_model(model, "nocaps_out", os.path.join(kwargs["nocaps_root"], "annotations/out-domain.json"), os.path.join(kwargs["nocaps_root"], "out-domain"))
    test_model(model, "nocaps_in", os.path.join(kwargs["nocaps_root"], "annotations/in-domain.json"), os.path.join(kwargs["nocaps_root"], "in-domain"))

    wandb.finish()

    return training_module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default="./checkpoints")
    parser.add_argument(
        "--annotations_file", default="./data/coco/annotations/captions_train2017.json"
    )
    parser.add_argument(
        "--val_annotations_file",
        default="./data/coco/annotations/captions_val2017.json",
    )
    parser.add_argument("--nocaps_root", default="./data/nocaps")
    parser.add_argument("--data_dir", default="./data/coco/train2017")
    parser.add_argument("--val_data_dir", default="./data/coco/val2017")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--prefix_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--mlp_hidden_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--find_lr", action="store_true")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--warmup_use_steps", action="store_true")
    parser.add_argument("--no_cosine", action="store_true")
    parser.add_argument("--use_unpooled_output", action="store_true")
    parser.add_argument(
        "--arch",
        default="flan-t5",
        choices=["mlp", "clipcap", "flan-t5", "flan-mlp", "flan-transformer"],
    )
    parser.add_argument(
        "--flan_size", default="small", choices=["small", "base", "large", "xl", "xxl"]
    )
    parser.add_argument("--gpt_size", default="", choices=["", "medium", "large", "xl"])
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--mlp_dropout", type=float, default=0.0)
    parser.add_argument(
        "--activation", type=str, default="tanh", choices=["tanh", "relu", "leaky"]
    )
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--finetune_lm", action="store_true")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--val_freq", type=int, default=None)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--direct_proj", action="store_true")

    args = parser.parse_args()

    if args.direct:
        args.prefix_length = 50
        args.finetune_lm = True
    if args.direct_proj:
        args.prefix_length = 50
    if args.lora:
        args.finetune_lm = False

    print("======= args =======")
    for k, v in vars(args).items():
        print(f"{k}: {v}")
    print("====================")

    train_model(**vars(args))


if __name__ == "__main__":
    main()
