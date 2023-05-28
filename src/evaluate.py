import torch
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

from PIL import Image
import os
import json


def evaluate(
    model,
    tokenizer,
    images,
    img_ids,
    tokens,
    arch: str = "mlp",
    answers_path: str = None,
):
    model.eval()
    references = {}
    hypothesis = {}
    device = next(model.parameters()).device
    cider = Cider()
    spice = Spice()

    hypothesis = {}
    references = {}
    answers = []

    for i in range(0, len(images), 128):
        with torch.no_grad():
            embeds = model.get_image_embeds(images[i : i + 128].to(device))
            _img_ids = img_ids[i : i + 128]
            _captions = generate(model, tokenizer, embeds, arch=arch)
            _decoded = tokenizer.batch_decode(
                tokens[i : i + 128], skip_special_tokens=True
            )
            for j in range(len(_captions)):
                id = _img_ids[j].cpu().item()
                if id not in hypothesis:
                    entry = {"caption": _captions[j], "image_id": id}
                    answers.append(entry)
                    hypothesis[id] = [entry]
                if id not in references:
                    references[id] = []
                references[id].append({"caption": _decoded[j], "image_id": id})
    if answers_path is not None:
        if not os.path.exists(answers_path):
            os.makedirs(answers_path)
        path = os.path.join(answers_path, "answers.json")
        with open(path, "w") as f:
            json.dump(answers, f)

    tokenizer = PTBTokenizer()
    hypothesis = tokenizer.tokenize(hypothesis)
    references = tokenizer.tokenize(references)

    cider_score, _ = cider.compute_score(references, hypothesis)
    spice_score, _ = spice.compute_score(references, hypothesis)

    return {"cider": float(cider_score) * 100, "spice": float(spice_score) * 100}


def generate(
    model,
    tokenizer,
    embed=None,
    entry_length=67,
    top_p=0.8,  # do we use top_p or temperature?
    temperature=1.0,
    stop_token: str = ".",  # might need to set dot for gpt2
    arch: str = "mlp",  # use 'lm_model'
):
    model.eval()
    batch_size = embed.size(0)
    generated = embed  # TODO: Conditional image "embed"s to encoder input, decoder to special token according to docs
    if (
        arch == "flan-t5"
        or arch == "flan-t5-trans"
        or arch == "flan-mlp"
        or arch == "flan-transformer"
    ):
        encoder_input = embed
        generated = model.token_to_embed(
            torch.zeros((batch_size, 1), dtype=torch.long).to(embed.device)
        )
        stop_token_id = tokenizer.eos_token_id
    else:
        stop_token_id = tokenizer.encode(stop_token)[0]
    eos_token_id = tokenizer.eos_token_id

    tokens = torch.zeros((batch_size, 0), dtype=torch.long).to(generated.device)

    with torch.no_grad():
        for i in range(entry_length):
            if (
                arch == "flan-t5"
                or arch == "flan-t5-trans"
                or arch == "flan-mlp"
                or arch == "flan-transformer"
            ):
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
            next_token = next_token.masked_fill(eos_mask, eos_token_id)
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

        tokens = tokens.squeeze().cpu()
        if len(tokens.shape) == 0:
            tokens = tokens.unsqueeze(0)
        if batch_size > 1:
            output_text = tokenizer.batch_decode(tokens, skip_special_tokens=True)
        else:
            output_text = tokenizer.decode(tokens, skip_special_tokens=True)
        return output_text
