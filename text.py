import torch
import pyopenjtalk
import re


def __fix_phone_tone(phone_tone_list: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """
    `phone_tone_list` の tone（アクセントの値）を 0 か 1 の範囲に修正する。
    例: [(a, 0), (i, -1), (u, -1)] → [(a, 1), (i, 0), (u, 0)]

    Args:
        phone_tone_list (list[tuple[str, int]]): 音素とアクセントのペアのリスト

    Returns:
        list[tuple[str, int]]: 修正された音素とアクセントのペアのリスト
    """

    tone_values = set(tone for _, tone in phone_tone_list)
    if len(tone_values) == 1:
        assert tone_values == {0}, tone_values
        return phone_tone_list
    elif len(tone_values) == 2:
        if tone_values == {0, 1}:
            return phone_tone_list
        elif tone_values == {-1, 0}:
            return [
                (letter, 0 if tone == -1 else 1) for letter, tone in phone_tone_list
            ]
        else:
            raise ValueError(f"Unexpected tone values: {tone_values}")
    else:
        raise ValueError(f"Unexpected tone values: {tone_values}")


def g2p_tone(text: str):
    def _numeric_feature_by_regex(regex: str, s: str) -> int:
        match = re.search(regex, s)
        if match is None:
            return -50
        return int(match.group(1))

    labels = pyopenjtalk.make_label(pyopenjtalk.run_frontend(text))
    N = len(labels)
    phones = []
    for n in range(N):
        lab_curr = labels[n]
        p3 = re.search(r"\-(.*?)\+", lab_curr).group(1)
        if p3 in "AEIOU":
            p3 = p3.lower()
        if p3 == "sil":
            assert n == 0 or n == N - 1
            if n == 0:
                phones.append("^")
            elif n == N - 1:
                # check question form or not
                e3 = _numeric_feature_by_regex(r"!(\d+)_", lab_curr)
                if e3 == 0:
                    phones.append("$")
                elif e3 == 1:
                    phones.append("?")
            continue
        elif p3 == "pau":
            phones.append("_")
            continue
        else:
            if p3 == "cl":
                p3 = "q"
            phones.append(p3)

        # accent type and position info (forward or backward)
        a1 = _numeric_feature_by_regex(r"/A:([0-9\-]+)\+", lab_curr)
        a2 = _numeric_feature_by_regex(r"\+(\d+)\+", lab_curr)
        a3 = _numeric_feature_by_regex(r"\+(\d+)/", lab_curr)

        # number of mora in accent phrase
        f1 = _numeric_feature_by_regex(r"/F:(\d+)_", lab_curr)

        a2_next = _numeric_feature_by_regex(r"\+(\d+)\+", labels[n + 1])
        # accent phrase border
        if a3 == 1 and a2_next == 1 and p3 in "aeiouAEIOUNq":
            phones.append("#")
        # pitch falling
        elif a1 == 0 and a2_next == a2 + 1 and a2 != f1:
            phones.append("]")
        # pitch rising
        elif a2 == 1 and a2_next == 2:
            phones.append("[")
    result: list[tuple[str, int]] = []
    current_phrase: list[tuple[str, int]] = []
    current_tone = 0

    for i, letter in enumerate(phones):
        # 特殊記号の処理

        # 文頭記号、無視する
        if letter == "^":
            assert i == 0, "Unexpected ^"
        # アクセント句の終わりに来る記号
        elif letter in ("$", "?", "_", "#"):
            # 保持しているフレーズを、アクセント数値を 0-1 に修正し結果に追加
            result.extend(__fix_phone_tone(current_phrase))
            # 末尾に来る終了記号、無視（文中の疑問文は `_` になる）
            if letter in ("$", "?"):
                assert i == len(phones) - 1, f"Unexpected {letter}"
            # あとは "_"（ポーズ）と "#"（アクセント句の境界）のみ
            # これらは残さず、次のアクセント句に備える。
            current_phrase = []
            # 0 を基準点にしてそこから上昇・下降する（負の場合は上の `fix_phone_tone` で直る）
            current_tone = 0
        # アクセント上昇記号
        elif letter == "[":
            current_tone = current_tone + 1
        # アクセント下降記号
        elif letter == "]":
            current_tone = current_tone - 1
        # それ以外は通常の音素
        else:
            current_phrase.append((letter, current_tone))
    text = [letter for letter, _ in result]
    tone = [tone + 1 for _, tone in result]
    return text, tone


symbols = [
    "_",
    "v",
    "N",
    "a",
    "a:",
    "b",
    "by",
    "ch",
    "d",
    "dy",
    "e",
    "e:",
    "f",
    "g",
    "gy",
    "h",
    "hy",
    "i",
    "i:",
    "j",
    "k",
    "ky",
    "m",
    "my",
    "n",
    "ny",
    "o",
    "o:",
    "p",
    "py",
    "q",
    "r",
    "ry",
    "s",
    "sh",
    "t",
    "ts",
    "ty",
    "u",
    "u:",
    "w",
    "y",
    "z",
    "zy",
]


def text_to_sequence(text: list[str]) -> torch.LongTensor:
    sequence = []
    for symbol in text:
        sequence.append(symbols.index(symbol))
    return torch.LongTensor(sequence)


def preprocess(text: str) -> tuple[torch.Tensor, list[int]]:
    phones, tones = g2p_tone(text)
    phones = text_to_sequence(phones)
    return phones, tones
