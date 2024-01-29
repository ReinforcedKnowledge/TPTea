import torch


def greedy_search(model, input_ids, max_length, eos_token_id):
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            # Check for EOS token
            if (
                eos_token_id is not None
                and torch.any(next_tokens == eos_token_id).item()
            ):
                break

            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

    return input_ids
