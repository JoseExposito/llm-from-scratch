import os
import torch


def generate_and_print_sample(model, config, start_context):
    """Dado un contexto inicial, genera un texto utilizando greedy decoding, es
    decir, eligiendo siempre el token más probable.
    """

    # Seleccionamos el dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    model.to(device)

    # Cargamos los pesos
    state_dict_path = os.path.join(
        os.getcwd(), "results-training", config.name, "weights.pth"
    )
    state_dict = torch.load(state_dict_path, map_location=device)
    model.load_state_dict(state_dict)

    # Generamos el texto
    model.eval()
    context_size = config.context_length
    encoded = _text_to_token_ids(start_context, config.tokenizer).to(device)
    endoftext_token = config.tokenizer.encode("<|endoftext|>")[0]

    with torch.no_grad():
        token_ids = _generate_text_greedy_decoding(
            model=model,
            idx=encoded,
            max_new_tokens=500,
            context_size=context_size,
            endoftext_token=endoftext_token,
        )
    decoded_text = _token_ids_to_text(token_ids, config.tokenizer)
    print("\nGenerated text:\n")
    print(decoded_text)
    model.train()


def _generate_text_greedy_decoding(
    model, idx, max_new_tokens, context_size, endoftext_token
):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        if idx_next.item() == endoftext_token:
            break

    return idx


def _text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Añade la dimensión del batch
    return encoded_tensor


def _token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # Quita la dimensión del batch
    return tokenizer.decode(flat.tolist())
