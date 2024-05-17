import torch
from fast import FastTTS
from fast_hparam import FastHparam
from matplotlib import pyplot as plt
from fast_preprocess import preprocess
import onnxruntime as ort
from scipy.io.wavfile import write
import time
import numpy as np
from librosa.display import waveshow


def load_model(
    hp: FastHparam, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
):
    model = FastTTS(hp)
    if hp.ckpt:
        model.load_state_dict(torch.load(hp.ckpt, map_location=device)["model"])
    model.to(device)
    model.eval()
    return model, device


def infer(
    model: FastTTS,
    string: str,
    device: torch.device,
):
    x, tones = preprocess(string)
    tones = torch.tensor(tones).unsqueeze(0).to(device)
    x_lengths = torch.tensor([len(x)]).to(device)
    with torch.no_grad():
        out = model.synthesise(
            x.unsqueeze(0).to(device),
            tones,
            x_lengths,
            20,
        )
    return out["decoder_outputs"].squeeze(0).cpu().numpy()


def main():
    hp = FastHparam()
    model, device = load_model(hp, "cpu")
    now = time.time()
    mel = infer(
        model,
        "ゆーちゅーぶは巨大な動画投稿プラットフォームです。そこでは様々な形式の動画が投稿され、その品質によってランク付けされ、視聴者に届けられます。",
        device,
    )
    print("Inference time:", time.time() - now)
    print(np.max(mel), np.min(mel))
    engine = ort.InferenceSession(
        "hifigan.onnx",
        providers=["CPUExecutionProvider"],
    )
    mel = mel.reshape(1, hp.num_mels, -1)
    audio: np.ndarray = engine.run(None, {"x": mel})[0]
    mel = mel[0]
    audio = audio[0]
    plt.figure()
    plt.subplots_adjust(hspace=0.5)
    plt.subplot(2, 1, 1)
    plt.title("Mel spectrogram")
    plt.imshow(mel)
    plt.subplot(2, 1, 2)
    plt.title("Waveform")
    waveshow(audio, sr=48000)
    plt.savefig("output.png")
    plt.close()
    write("output.wav", 48000, audio.T)


if __name__ == "__main__":
    main()
