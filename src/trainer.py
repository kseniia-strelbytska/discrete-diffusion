import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime

from loss import rblb
from evaluation_tools import evaluation_from_generation
from transformers.optimization import get_inverse_sqrt_schedule
import wandb


def train(
    model,
    grammar,
    device,
    T=500,
    eos_weight=1.0,
    inverse_t=False,
    dirs=None,
    evaluation_config=None,
    validation_config=None,
    epochs=5,
    lr=1e-3,
    num_warmup_steps=1000,
    weight_decay=0.01,
    train_dataloader=None,
    test_dataset=None,
    evaluation_dataset=None,
    verbose=False,
    save_mode: bool = False,
    strategy='categorical',
    wandb=None,
    loss_type="eq8",
    denoise="0"
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if num_warmup_steps != 0:
        lr_scheduler = get_inverse_sqrt_schedule(
            optimizer, num_warmup_steps=num_warmup_steps
        )
    loss_fn = rblb(device, vocab_size=model.vocab_size, T=T, sampling_eps=model.sampling_eps, eos_weight=eos_weight, inverse_t=inverse_t, loss_type=loss_type)

    stats = [[], [], [], [], []]  # r1, r2, both, format, epochsteps
    test_loss_stats, train_loss_stats = [], []
    test_loss_epochs = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Optionally also log header to file if saving is enabled.
    if save_mode:
        header_line = "-" * 20 + f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        with open(dirs.loss_log_path, "a") as f:
            f.write(header_line + "\n")

    epochs_iter = (
        range(epochs) if verbose else tqdm(range(epochs), desc="Training Epochs")
    )

    for epoch in epochs_iter:
        #Calculate the eval metrics at initialisation
        if epoch == 0 and False: # skip initial evaluation to save time; set to True to enable
            new_stats, new_stats_eos, total_eos, sequences, sequences_eos = evaluation_from_generation(
                model,
                grammar,
                evaluation_dataset=evaluation_dataset,
                T=T,
                strategy = strategy,
                write_steps=False,
                device=device,
                figures_path=dirs.figure_path,
                loss_log_path=dirs.loss_log_path,
                output_path=dirs.output_path,
                save_mode=save_mode,
                denoise=denoise
            )
            for i in range(4):
                stats[i].append(new_stats[i])
            stats[-1].append(epoch + 1)

            # Log evaluation stats to Weights & Biases if enabled.
            panel_name = "Rule_Accuracy"
            if wandb.run is not None:
                table = wandb.Table(columns=["sequence"])
                for seq in sequences:
                    table.add_data(seq)
                table_eos = wandb.Table(columns=["sequence"])
                for seq in sequences_eos:
                    table_eos.add_data(seq)
                wandb.log(
                    {
                        f'{panel_name}/epoch': epoch + 1,
                        f'{panel_name}/eval_rule1_acc': float(new_stats[0]),
                        f'{panel_name}/eval_rule2_acc': float(new_stats[1]),
                        f'{panel_name}/eval_both_rules_acc': float(new_stats[2]),
                        f'{panel_name}/eval_format_acc': float(new_stats[3]),
                        f'{panel_name}/generated_sequences': table,
                        f'{panel_name}_Finished/epoch': epoch + 1,
                        f'{panel_name}_Finished/finished_pct': total_eos / len(sequences),
                        f'{panel_name}_Finished/eval_rule1_acc': float(new_stats_eos[0]),
                        f'{panel_name}_Finished/eval_rule2_acc': float(new_stats_eos[1]),
                        f'{panel_name}_Finished/eval_both_rules_acc': float(new_stats_eos[2]),
                        f'{panel_name}_Finished/eval_format_acc': float(new_stats_eos[3]),
                        f'{panel_name}_Finished/generated_sequences': table_eos,
                    },
                    step=epoch + 1,
                )

        #Train the model
        total_loss = 0
        for x_batch, y_batch, timestep in train_dataloader:
            optimizer.zero_grad()
            logits = model(x_batch, timestep)
            loss = loss_fn(x_batch, logits, y_batch, timestep)
            loss.backward()
            optimizer.step()
            if num_warmup_steps != 0:
                lr_scheduler.step()

            total_loss += loss.item()

        if epoch % validation_config.val_every == 0:
            # Optimization (train-mode) loss averaged over the epoch.
            avg_train_loss = total_loss / len(train_dataloader)
            train_loss_stats.append(avg_train_loss)

            model.eval()
            with torch.no_grad():
                # Test loss in eval mode on the held-out loader.
                test_loss = 0.0
                for X_batch, y_batch, timestep in test_dataset:
                    logits = model(X_batch, timestep)
                    loss = loss_fn(X_batch, logits, y_batch, timestep)
                    test_loss += loss.item()
                avg_test_loss = test_loss / len(test_dataset)
                test_loss_stats.append(avg_test_loss)
                test_loss_epochs.append(epoch + 1)

            # IMPORTANT: switch model back to train mode after validation
            model.train()

            current_lr = lr
            if num_warmup_steps != 0:
                current_lr = lr_scheduler.get_last_lr()[0]

            log_line = (
                f"Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Test Loss: {avg_test_loss:.4f}, "
                f"Current lr: {current_lr}"
            )

            print(log_line)

            # Log to Weights & Biases if enabled.
            panel_name = "Loss"
            if wandb.run is not None:
                wandb.log(
                    {
                        f'{panel_name}/epoch': epoch + 1,
                        f'{panel_name}/train_loss': avg_train_loss,
                        f'{panel_name}/test_loss': avg_test_loss,
                    },
                    step=epoch + 1,
                )

            # Optionally append to the loss log file if saving is enabled.
            if save_mode and dirs is not None:
                with open(dirs.loss_log_path, "a") as f:
                    f.write(log_line + "\n")

        if (epoch + 1) % evaluation_config.eval_every == 0:
            new_stats, new_stats_eos, total_eos, sequences, sequences_eos = evaluation_from_generation(
                model,
                grammar,
                evaluation_dataset=evaluation_dataset,
                T=T,
                strategy = strategy,
                write_steps=False,
                device=device,
                figures_path=dirs.figure_path,
                loss_log_path=dirs.loss_log_path,
                output_path=dirs.output_path,
                save_mode=save_mode,
                denoise=denoise
            )
            for i in range(4):
                stats[i].append(new_stats[i])
            stats[-1].append(epoch + 1)

            # Log evaluation stats to Weights & Biases if enabled.
            panel_name = "Rule_Accuracy"
            if wandb.run is not None:
                table = wandb.Table(columns=["sequence"])
                for seq in sequences:
                    table.add_data(seq)
                table_eos = wandb.Table(columns=["sequence"])
                for seq in sequences_eos:
                    table_eos.add_data(seq)
                wandb.log(
                    {
                        f'{panel_name}/epoch': epoch + 1,
                        f'{panel_name}/eval_rule1_acc': float(new_stats[0]),
                        f'{panel_name}/eval_rule2_acc': float(new_stats[1]),
                        f'{panel_name}/eval_both_rules_acc': float(new_stats[2]),
                        f'{panel_name}/eval_format_acc': float(new_stats[3]),
                        f'{panel_name}/generated_sequences': table,
                        f'{panel_name}_Finished/epoch': epoch + 1,
                        f'{panel_name}_Finished/finished_pct': total_eos / len(sequences),
                        f'{panel_name}_Finished/eval_rule1_acc': float(new_stats_eos[0]),
                        f'{panel_name}_Finished/eval_rule2_acc': float(new_stats_eos[1]),
                        f'{panel_name}_Finished/eval_both_rules_acc': float(new_stats_eos[2]),
                        f'{panel_name}_Finished/eval_format_acc': float(new_stats_eos[3]),
                        f'{panel_name}_Finished/generated_sequences': table_eos,
                    },
                    step=epoch + 1,
                )

            ax1.clear()
            ax1.plot(test_loss_epochs, train_loss_stats)
            ax1.plot(test_loss_epochs, test_loss_stats)
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend(["Train Loss", "Test Loss"], loc="lower right")
            ax1.set_title("Loss vs Epoch")
            ax1.grid(True)

            ax2.clear()
            ax2.plot(stats[-1], stats[0])
            ax2.plot(stats[-1], stats[1])
            ax2.plot(stats[-1], stats[2])
            ax2.plot(stats[-1], stats[3])
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend(["Rule 1", "Rule 2", "Both Rules", "Format"], loc="lower right")

            plt.tight_layout()
            
            # Save plot and checkpoint only when saving is enabled.
            if save_mode and dirs is not None:
                plt.savefig(dirs.figure_path / "plot.png", dpi=150)
                torch.save(
                    model.state_dict(), dirs.model_path / f"model_epochs={epoch + 1}"
                )

    return model