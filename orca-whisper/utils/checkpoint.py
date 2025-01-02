import os
import torch


def initialize_best_dict():
    return {
        "generator": {
            "model_state_dict": None,
            "iteration": None,
            "loss": float('inf'),
            "samples": None
        },
        "discriminator": {
            "model_state_dict": None,
            "iteration": None,
            "loss": float('inf')
        }
    }


def restore_checkpoint(training_directory, generator, discriminator, device, checkpoint_number=None):
    iteration = 0
    d_losses = []
    g_losses = []
    checkpoints = [ckp for ckp in os.listdir(training_directory) if
                   os.path.isfile(os.path.join(training_directory, ckp)) and
                   os.path.splitext(os.path.join(training_directory, ckp))[1].lower() == '.ckp']
    best_dict = initialize_best_dict()
    ckp_number = -1
    if len(checkpoints) != 0:
        checkpoint_numbers = [int(checkpoint.split('-')[1].replace('.ckp', '')) for checkpoint in checkpoints]
        ckp_number = checkpoint_number or max(checkpoint_numbers)
        iteration = ckp_number
        checkpoint = os.path.join(training_directory, 'ckp-{}.ckp'.format(ckp_number))
        checkpoint = torch.load(checkpoint)

        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        d_losses = checkpoint['d_losses']
        g_losses = checkpoint['g_losses']
        best_dict = best_dict if 'best' not in checkpoint else checkpoint['best']

    generator.to(device)
    discriminator.to(device)

    return {
        'iteration': iteration,
        'd_losses': d_losses,
        'g_losses': g_losses,
        'best': best_dict

    }, ckp_number


def get_checkpoints(ckp_dir):
    return sorted([os.path.join(ckp_dir, c) for c in os.listdir(ckp_dir) if c.endswith(".ckp")],
                  key=lambda x: int(os.path.basename(x).replace("ckp-", "").replace(".ckp", "")))


def save_checkpoint(generator,
                    discriminator,
                    checkpoint_directory,
                    checkpoint_number=None,
                    g_losses=None,
                    d_losses=None,
                    max_checkpoints=10,
                    best_dict=None,
                    final=False,
                    class_name=None):

    if not final:
        checkpoint = os.path.join(checkpoint_directory, 'ckp-{}.ckp'.format(checkpoint_number))
        torch.save(
            {'generator_state_dict': generator.state_dict(),
             'discriminator_state_dict': discriminator.state_dict(),
             'd_losses': d_losses,
             'g_losses': g_losses,
             'best': best_dict
             }, checkpoint)

        checkpoints = get_checkpoints(checkpoint_directory)

        if len(checkpoints) > max_checkpoints:
            to_delete = checkpoints[0:len(checkpoints) - max_checkpoints]
            for ckp in to_delete:
                os.remove(ckp)
    else:
        checkpoint = os.path.join(checkpoint_directory, f'ORCA-WHISPER-{class_name}.pt')
        torch.save(
            {'generator_state_dict': generator.state_dict(),
             'discriminator_state_dict': discriminator.state_dict()
             },
            checkpoint)