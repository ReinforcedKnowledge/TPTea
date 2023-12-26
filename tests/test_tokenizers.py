import json
import tempfile

from src.tokenizers.bpe import BPE, BPEConfig
from src.tokenizers.utils import InvertibleDict, InvertibleDictEncoder

VOCAB_SIZE = 13  # ~ to 13-7 = 6 merges

cfg = BPEConfig()
cfg.base_vocab = "bghnpsu"
cfg.vocab_size = VOCAB_SIZE
corpus = "hug " * 10 + "pug " * 5 + "pun " * 12 + "bun " * 4 + "hugs " * 5

bpe = BPE(cfg)
bpe.train(corpus)

expected_vocab = InvertibleDict(
    {
        "<unk>": 0,
        "<pad>": 1,
        "<bos>": 2,
        "<eos>": 3,
        "b": 4,
        "g": 5,
        "h": 6,
        "n": 7,
        "p": 8,
        "s": 9,
        "u": 10,
        "ug": 11,
        "un": 12,
    }
)


def test_bpe_tokenizer():
    learned_vocab = bpe.get_vocab
    assert learned_vocab == expected_vocab


def test_save_vocab():
    # use a temp file to avoid writing to disk
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        bpe.save(temp_file.name, overwrite=True)

        temp_file.seek(0)
        saved_vocab = json.load(temp_file)
        assert saved_vocab == bpe.get_vocab


def test_load_vocab():
    # use a temp file to avoid writing to disk
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        json.dump(expected_vocab, temp_file, cls=InvertibleDictEncoder)
        temp_file_name = temp_file.name

    bpe.load(temp_file_name)
    assert bpe.get_vocab == expected_vocab
