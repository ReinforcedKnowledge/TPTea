from src.tokenizers.bpe import BPE, BPEConfig
import json
import tempfile

VOCAB_SIZE = 13  # ~ to 13-7 = 6 merges

cfg = BPEConfig()
cfg.base_vocab = "bghnpsu"
cfg.vocab_size = VOCAB_SIZE
corpus = "hug " * 10 + "pug " * 5 + "pun " * 12 + "bun " * 4 + "hugs " * 5

bpe = BPE(cfg)
bpe.train(corpus)

expected_vocab = {
    "b": 0,
    "g": 1,
    "h": 2,
    "n": 3,
    "p": 4,
    "s": 5,
    "u": 6,
    "ug": 7,
    "un": 8,
    "hug": 9,
    "pun": 10,
    "pug": 11,
    "hugs": 12,
}


def test_bpe_tokenizer():
    learned_vocab = bpe.get_vocab()
    assert learned_vocab == expected_vocab


def test_save_vocab():
    # use a temp file to avoid writing to disk
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        bpe.save(tempfile.name)

        temp_file.seek(0)
        saved_vocab = json.load(temp_file)
        assert saved_vocab == bpe.get_vocab()


def test_load_vocab():
    # use a temp file to avoid writing to disk
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
        json.dump(expected_vocab, temp_file)

        bpe.load(tempfile.name)
        assert bpe.get_vocab() == expected_vocab
