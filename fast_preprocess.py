from fast_hparam import FastHparam
from mel import load_wav, mel_spectrogram
import torch
from text import preprocess
from tqdm import tqdm
from transformers import BertTokenizerFast
import onnxruntime as ort
from huggingface_hub import hf_hub_download


def load_bert_model():
    bert_tok = BertTokenizerFast.from_pretrained("neody/ja-bert-1")
    bert_model = ort.InferenceSession(
        hf_hub_download("neody/ja-bert-1", "hidden_model.onnx"),
        providers=["CUDAExecutionProvider"],
    )
    return bert_tok, bert_model


def bert_fit(b: torch.Tensor, l: int):
    # [x, 768] -> [l, 768]
    if l < b.shape[0]:
        return b[:l]
    bn = torch.zeros(l, b.shape[1])
    rest = l % b.shape[0]
    arg = (l - rest) // b.shape[0]
    for i in range(b.shape[0]):
        if i < rest:
            bn[i : i + arg + 1] = b[i].repeat(arg + 1, 1)
        else:
            bn[i : i + arg] = b[i].repeat(arg, 1)
    return bn


def preprocess_all(hp: FastHparam):
    with open(hp.train_file, "r", encoding="utf8") as r:
        r = r.readlines()
        for ix, line in enumerate(tqdm(r)):
            f, i, c = line.strip().split("|")
            f = load_wav(f"{hp.audio_root}/{f}", hp.sample_rate)
            mel = mel_spectrogram(
                torch.from_numpy(f).unsqueeze(0),
                hp.n_fft,
                hp.num_mels,
                hp.sample_rate,
                hp.hop_size,
                hp.win_size,
                hp.fmin,
                hp.fmax,
                center=False,
            ).squeeze()
            c, t = preprocess(c)
            if len(t) > 512 or len(t) < 5:
                continue
            t = torch.tensor(t, dtype=torch.long)
            s = (c, t, mel, int(i))
            torch.save(s, f"{hp.train_set}/{ix + 1}.pt")


if __name__ == "__main__":
    hp = FastHparam()
    preprocess_all(hp)
