from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from fast_hparam import FastHparam
from fast_train import Dataset


def compute_data_statistics(data_loader: DataLoader, out_channels: int):
    """Generate data mean and standard deviation helpful in data normalisation

    Args:
        data_loader (torch.utils.data.Dataloader): _description_
        out_channels (int): mel spectrogram channels
    """
    total_mel_sum = 0
    total_mel_sq_sum = 0
    total_mel_len = 0

    for batch in tqdm(data_loader, leave=False):
        text, tone, text_tone_lens, mels, mel_lengths, speaker_id = batch

        total_mel_len += torch.sum(mel_lengths)
        total_mel_sum += torch.sum(mels)
        total_mel_sq_sum += torch.sum(torch.pow(mels, 2))

    data_mean = total_mel_sum / (total_mel_len * out_channels)
    data_std = torch.sqrt(
        (total_mel_sq_sum / (total_mel_len * out_channels)) - torch.pow(data_mean, 2)
    )

    return {"mel_mean": data_mean.item(), "mel_std": data_std.item()}


if __name__ == "__main__":
    hp = FastHparam()
    dataset = Dataset(hp)
    data_loader = DataLoader(
        dataset,
        batch_size=hp.batch_size,
        num_workers=hp.threads,
        shuffle=hp.shuffle,
        collate_fn=dataset.collate_fn,
    )
    data_stats = compute_data_statistics(data_loader, hp.num_mels)
    print(data_stats)
