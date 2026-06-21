import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import time
import torch

from llm_from_scratch.data_loader import create_dataloader


def train_model(model, config):
    """Entrena el modelo con el corpus TinyStories."""

    # Creamos el directorio donde se guardarán los resultados del entrenamiento
    results_dir = os.path.join(os.getcwd(), "results-training", config.name)
    if os.path.exists(results_dir):
        raise Exception(
            f"Path '{results_dir}' already exists, delete it to re-train the model"
        )
    os.makedirs(results_dir)

    # Entrenar el modelo lleva unas 6 horas en una GPU, si no está disponible,
    # detenemos el entrenamiento. En una CPU es demasiado costoso
    if not torch.cuda.is_available():
        raise Exception("Cuda is not available")

    device = "cuda"
    print(f"Using {device} device")
    model.to(device)

    # Usaremos AdamW como optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    # Preparamos los datos de entrenamiento y validación
    print(f"Setting up data loaders ({time.asctime()})")
    train_loader = create_dataloader(
        tokenizer=config.tokenizer,
        split="train",
        batch_size=32,
        window_size=config.context_length,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader(
        tokenizer=config.tokenizer,
        split="validation",
        batch_size=32,
        window_size=config.context_length,
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    # Entrenamos el modelo
    print(f"Training started ({time.asctime()})")
    num_epochs = 1
    train_losses, val_losses, tokens_seen = _training_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=500,
        eval_iter=5,
    )
    print(f"Training finished ({time.asctime()})")

    # Guardamos los resultados
    torch.save(model.state_dict(), os.path.join(results_dir, "weights.pth"))
    _plot_losses(results_dir, num_epochs, tokens_seen, train_losses, val_losses)


def _training_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reseteamos los loss gradients de la ejecución anterior
            loss = _calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # Calculamos los loss gradients
            optimizer.step()  # Actualizamos los pesos del modelo usando los loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = _evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}, "
                    f"Perplexity {torch.exp(torch.tensor(train_loss)).item():.3f}"
                )

    return train_losses, val_losses, track_tokens_seen


def _calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def _calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = _calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def _evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = _calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = _calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def _plot_losses(results_dir, num_epochs, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    epochs_seen = torch.linspace(0, num_epochs, len(train_losses))

    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.savefig(os.path.join(results_dir, "loss-plot.png"))
