import torch
import torch.nn.functional as nnf
from pycocotools.coco import COCO
import pycocoevalcap
from pycocoevalcap.cider.cider import Cider

# from pycocoevalcap.spice.spice import Spice
import PIL
from PIL import Image
import io
import os


# todo: optimize it, generate captions in batches
def evaluate(model, tokenizer, images, tokens, arch: str = "mlp"):
    model.eval()
    references = {}
    hypothesis = {}
    device = next(model.parameters()).device
    cider = Cider()
    # spice = Spice()

    with torch.no_grad():
        embeds = model.get_image_embeds(images.to(device))
        captions = generate(model, tokenizer, embeds, arch=arch)

    if tokens.shape[0] > 1:
        decoded_tokens = tokenizer.batch_decode(tokens, skip_special_tokens=True)
    else:
        decoded_tokens = tokenizer.decode(tokens, skip_special_tokens=True)

    hypothesis = {i + 1: [caption] for i, caption in enumerate(captions)}
    references = {i + 1: [token] for i, token in enumerate(decoded_tokens)}

    cider_score, _ = cider.compute_score(references, hypothesis)
    # spice_score, _ = spice.compute_score(references, hypothesis)

    return {
        "cider": float(cider_score) * 100,
        # 'spice': spice_score
    }


# todo: optimize it, generate captions in batches
def generate(
    model,
    tokenizer,
    embed=None,
    entry_length=50,
    top_p=0.8,  # do we use top_p or temperature?
    temperature=1.0,
    # stop_token: str = ".", # might need to set dot for gpt2
    arch: str = "mlp",  # use 'lm_model'
):
    model.eval()
    stop_token_id = tokenizer.eos_token_id
    batch_size = embed.size(0)
    generated = embed  # TODO: Conditional image "embed"s to encoder input, decoder to special token according to docs
    if arch == "flan-t5" or arch == "flan-mlp" or arch == "flan-transformer":
        encoder_input = embed
        generated = model.token_to_embed(
            torch.zeros((batch_size, 1), dtype=torch.long).to(embed.device)
        )

    tokens = torch.zeros((batch_size, 0), dtype=torch.long).to(generated.device)

    with torch.no_grad():
        for i in range(entry_length):
            if arch == "flan-t5" or arch == "flan-mlp" or arch == "flan-transformer":
                logits = model.get_logits(encoder_input, generated)
            else:
                logits = model.get_logits(generated)
            logits = logits[:, -1, :]
            eos_mask = torch.any(tokens == stop_token_id, dim=1)
            eos_mask = eos_mask.unsqueeze(1)
            next_token = torch.argmax(logits, -1).squeeze(
                0
            )  # remove the extra dimension
            if batch_size > 1:
                next_token = next_token.unsqueeze(1)  # add the batch dimension
            next_token = next_token.masked_fill(eos_mask, stop_token_id)
            next_token_embed = model.token_to_embed(next_token)
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            generated = torch.cat(
                (generated, next_token_embed), dim=1
            )  # Concatenate along the sequence dimension
            if torch.all(torch.any(tokens == stop_token_id, dim=1)):
                break

        tokens = tokens.squeeze().cpu()  # TODO: Why is this done on cpu?
        if len(tokens.shape) == 0:
            tokens = tokens.unsqueeze(0)
        if batch_size > 1:
            output_text = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        else:
            output_text = tokenizer.decode(tokens, skip_special_tokens=True)
        return output_text
