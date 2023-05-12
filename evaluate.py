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
def evaluate(
    model, tokenizer, images, tokens):
    model.eval()
    references = {}
    hypothesis = {}
    device = next(model.parameters()).device
    cider = Cider()
    # spice = Spice()

    with torch.no_grad():
        # (batch_size, dim1, dim2)
        embeds = model.get_image_embeds(images.to(device))
        # (1, dim1, dim2)
        embeds = torch.split(embeds, 1)
        # (dim1, dim2)
        for i, image_embeds in enumerate(embeds):
            caption = generate(model, tokenizer, image_embeds)
            hypothesis[i] = [caption]
            references[i] = [tokenizer.decode(tokens[i].cpu().numpy())]

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
    stop_token: str = ".",
):
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    generated = embed
    tokens = None

    with torch.no_grad():
        for i in range(entry_length):
            logits = model.get_logits(generated)
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            # is it a bottleneck?
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(nnf.softmax(sorted_logits, dim=-1), dim=-1)
            # use top_p conditionally?
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.token_to_embed(next_token)
            if tokens is None:
                tokens = next_token
            else:
                tokens = torch.cat((tokens, next_token), dim=1)
            generated = torch.cat((generated, next_token_embed), dim=1)
            if stop_token_index == next_token.item():
                break

        tokens = tokens.squeeze().cpu()
        if len(tokens.shape) == 0:
            tokens = tokens.unsqueeze(0)
        output_list = list(tokens.numpy())
        output_text = tokenizer.decode(output_list)
        return output_text