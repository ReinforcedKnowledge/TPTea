from src.tokenizers.bpe import BPE, BPEConfig


def tokenize_dummy_text():
    cfg = BPEConfig()
    bpe = BPE(cfg)
    corpus = "hug " * 10 + "pug " * 5 + "pun " * 12 + "bun " * 4 + "hugs " * 5
    bpe.train(corpus)

    return bpe.vocab


def test_bpe_tokenizer():
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
    assert tokenize_dummy_text() == expected_vocab
