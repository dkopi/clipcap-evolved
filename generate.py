import torch
import torch.nn.functional as nnf

# optimize it
def generate(
    model,
    tokenizer,
    embed=None,
    entry_length=50,
    top_p=0.8, # do we use top_p or temperature
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    generated_num = 0
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device
    generated = embed
    tokens = None

    with torch.no_grad():
        for i in range(entry_length):
            outputs = model(inputs_embeds=generated)
            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                nnf.softmax(sorted_logits, dim=-1), dim=-1
            )
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                ..., :-1
            ].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = filter_value
            next_token = torch.argmax(logits, -1).unsqueeze(0)
            next_token_embed = model.transformer.wte(next_token)
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
