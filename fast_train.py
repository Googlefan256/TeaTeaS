from torch import optim
from tqdm import tqdm
from common import uncompile
from fast_hparam import FastHparam
import torch
from fast import FastTTS
from torch.utils.data import Dataset as TDS, DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import os

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class Dataset(TDS):
    def __init__(self, hp: FastHparam):
        super(Dataset, self).__init__()
        self.hp = hp
        self.datas = os.listdir(hp.train_set)
        self.datas = [x for x in self.datas if x.endswith(".pt")]
        print(f"Found {len(self.datas)} files")

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, item):
        # text[N], tone[N], mel[n_mel, O], speaker_id
        return torch.load(f"{self.hp.train_set}/{self.datas[item]}")

    def collate_fn(self, batch):
        batch = sorted(batch, key=lambda x: x[1].shape[0])
        text, tone, mel, speaker_id = zip(*batch)
        text_tone_longest = max([x.shape[0] for x in text])
        mel_longest = max([x.shape[1] for x in mel])
        if mel_longest % 2 != 0:
            mel_longest += 1
        text_tone_lens = torch.tensor([x.shape[0] for x in text], dtype=torch.long)
        text = torch.stack(
            [F.pad(x, (0, text_tone_longest - x.shape[0])) for x in text]
        )
        tone = torch.stack(
            [F.pad(x, (0, text_tone_longest - x.shape[0])) for x in tone]
        )
        mel_lens = torch.tensor([x.shape[1] for x in mel], dtype=torch.long)
        mel = torch.stack([F.pad(x, (0, mel_longest - x.shape[1])) for x in mel])
        speaker_id = torch.tensor(speaker_id, dtype=torch.long)
        return text, tone, text_tone_lens, mel, mel_lens, speaker_id


def train(hp: FastHparam):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FastTTS(hp)
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    epoch = 1
    steps = 0
    if hp.ckpt:
        ckpt = torch.load(hp.ckpt)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        epoch = ckpt["epoch"]
        steps = ckpt["steps"]
    if hp.compile_mod:
        model = torch.compile(model)
    scheduler = optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=hp.lr_decay, last_epoch=-1 if epoch == 1 else epoch
    )
    dataset = Dataset(hp)
    dataloader = DataLoader(
        dataset,
        batch_size=hp.batch_size,
        shuffle=hp.shuffle,
        collate_fn=dataset.collate_fn,
        num_workers=hp.threads,
    )
    sw = SummaryWriter("logs")
    for _ in range(epoch, 100_000_000):
        loss_total = 0
        loss_avg = 0
        for i, (text, tone, text_tone_lens, mel, mel_lens, speaker_id) in tqdm(
            enumerate(dataloader)
        ):
            optimizer.zero_grad()
            text = text.to(device)
            tone = tone.to(device)
            text_tone_lens = text_tone_lens.to(device)
            mel = mel.to(device)
            mel_lens = mel_lens.to(device)
            speaker_id = speaker_id.to(device)
            dur, prior, diff = model(
                text,
                tone,
                speaker_id,
                text_tone_lens,
                mel,
                mel_lens,
            )
            loss = dur + prior + diff
            loss.backward()
            clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_total += loss.item()
            loss_avg = loss_total / (i + 1)
            steps += 1
            if steps % 10 == 0:
                sw.add_scalar("fast/loss", loss.item(), steps)
                sw.add_scalar("fast/prior", prior.item(), steps)
                sw.add_scalar("fast/diff", diff.item(), steps)
                sw.add_scalar("fast/dur", dur.item(), steps)
            if steps % hp.save_steps == 0 and steps != 0:
                torch.save(
                    {
                        "model": uncompile(model.state_dict()),
                        "optimizer": uncompile(optimizer.state_dict()),
                        "epoch": epoch + 1,
                        "steps": steps,
                    },
                    f"output/fast_{epoch}_{steps}.pt",
                )
        print(f"Epoch {epoch} Loss: {loss_avg}")
        scheduler.step()
        epoch += 1


if __name__ == "__main__":
    hp = FastHparam()
    train(hp)
