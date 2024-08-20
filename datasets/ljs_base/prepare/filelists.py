import os
import sys
import pandas as pd
import sys

curr_dir = os.getcwd().split('/')
vits_path = '/'.join(curr_dir)
from utils.hparams import get_hparams_from_file
# See: https://github.com/espeak-ng/espeak-ng/blob/master/docs/languages.md

dir_data = "./datasets-vits2"
config = vits_path + "/datasets/ljs_base/config.yaml"
symlink = "DUMMY1"
n_val = 100
n_test = 500

hps = get_hparams_from_file(config)



data = pd.read_csv(
    f"{dir_data}/metadata_copy.csv",
    sep=r"|",
    header=None,
    names=["file", "text", "normalized_text", "cleaned_text"],
    index_col=False,
    # converter to add .wav to file name
    converters={"file": lambda x: f"{symlink}/{x.strip()}.wav", "text": str.strip, "normalized_text": str.strip},
)
data.head()


# Get index of tokenize_text
text_cleaners = hps.data.text_cleaners

token_idx = text_cleaners.index("tokenize_text")
token_cleaners = text_cleaners[token_idx:]
print(token_cleaners)


# Extract phonemize_text
def separate_text_cleaners(text_cleaners):
    final_list = []
    temp_list = []

    for cleaner in text_cleaners:
        if cleaner == "phonemize_text":
            if temp_list:
                final_list.append(temp_list)
            final_list.append([cleaner])
            temp_list = []
        else:
            temp_list.append(cleaner)

    if temp_list:
        final_list.append(temp_list)

    return final_list


text_cleaners = text_cleaners[:token_idx]
text_cleaners = separate_text_cleaners(text_cleaners)
print(text_cleaners)


from text import tokenizer
from torchtext.vocab import Vocab

text_norm = data["normalized_text"].tolist()
for cleaners in text_cleaners:
    print(f"Cleaning with {cleaners} ...")
    if cleaners[0] == "phonemize_text":
        text_norm = tokenizer(text_norm, Vocab, cleaners, language=hps.data.language)
    else:
        for idx, text in enumerate(text_norm):
            temp = tokenizer(text, Vocab, cleaners, language=hps.data.language)
            text_norm[idx] = temp

data = data.assign(cleaned_text=text_norm)
data.head()


from torchtext.vocab import build_vocab_from_iterator
from utils.task import load_vocab, save_vocab
from text.symbols import special_symbols, UNK_ID
from typing import List


def yield_tokens(cleaned_text: List[str]):
    for text in cleaned_text:
        yield text.split()


text_norm = data["cleaned_text"].tolist()
vocab = build_vocab_from_iterator(yield_tokens(text_norm), specials=special_symbols)
vocab.set_default_index(UNK_ID)

vocab_file = f"../vocab.txt"
save_vocab(vocab, vocab_file)

vocab = load_vocab(vocab_file)
print(f"Size of vocabulary: {len(vocab)}")
print(vocab.get_itos())


from text import detokenizer

text_norm = data["cleaned_text"].tolist()
for idx, text in enumerate(text_norm):
    temp = tokenizer(text, vocab, token_cleaners, language=hps.data.language)
    assert UNK_ID not in temp, f"Found unknown symbol:\n{text}\n{detokenizer(temp)}"
    text_norm[idx] = temp

text_norm = ["\t".join(map(str, text)) for text in text_norm]
data = data.assign(tokens=text_norm)
data.head()



data = data[["file", "tokens"]]
data = data.sample(frac=1).reset_index(drop=True)

data_train = data.iloc[n_val + n_test:]
data_val = data.iloc[:n_val]
data_test = data.iloc[n_val: n_val + n_test]

data_train.to_csv("../filelists/train.txt", sep="|", index=False, header=False)
data_val.to_csv("../filelists/val.txt", sep="|", index=False, header=False)
data_test.to_csv("../filelists/test.txt", sep="|", index=False, header=False)