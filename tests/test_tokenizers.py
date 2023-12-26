import json
import tempfile

from src.tokenizers.bpe import BPE, BPEConfig
from src.tokenizers.utils import InvertibleDict, InvertibleDictEncoder

VOCAB_SIZE = 16  # ~ to 16-12 = 4 merges

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
        "Ġ": 4,
        "b": 5,
        "g": 6,
        "h": 7,
        "n": 8,
        "p": 9,
        "s": 10,
        "u": 11,
        "ug": 12,
        "Ġp": 13,
        "un": 14,
        "hug": 15,
    }
)


def test_bpe_trainer():
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


def test_break_into_subwords():
    texts = ["bug", "mug", "bug mug hug hugs!"]
    expected_tokens = [
        ["b", "ug"],
        ["m", "ug"],
        ["b", "ug", "Ġ", "m", "ug", "Ġ", "hug", "Ġ", "hug", "s", "!"],
    ]

    tokens = [bpe.break_into_subwords(bpe.pre_tokenize(text)) for text in texts]

    assert tokens == expected_tokens


def test_tokenize():
    texts = ["bug", "mug", "bug mug hug hugs!"]
    expected_tokens = [
        ["<bos>", "b", "ug", "<eos>"],
        ["<bos>", "<unk>", "ug", "<eos>"],
        [
            "<bos>",
            "b",
            "ug",
            "Ġ",
            "<unk>",
            "ug",
            "Ġ",
            "hug",
            "Ġ",
            "hug",
            "s",
            "<unk>",
            "<eos>",
        ],
    ]

    token_ids = [bpe.tokenize(text) for text in texts]
    expected_token_ids = [[bpe.vocab[t] for t in tokens] for tokens in expected_tokens]

    assert token_ids == expected_token_ids


def test_detokenize():
    token_sequences = [
        ["<bos>", "b", "ug", "<eos>"],
        ["<bos>", "<unk>", "ug", "<eos>"],
        [
            "<bos>",
            "b",
            "ug",
            "Ġ",
            "<unk>",
            "ug",
            "Ġ",
            "hug",
            "Ġ",
            "hug",
            "s",
            "<unk>",
            "<eos>",
        ],
    ]
    expected_texts = ["bug", "<unk>ug", "bug <unk>ug hug hugs<unk>"]

    token_id_sequences = [[bpe.vocab[t] for t in tokens] for tokens in token_sequences]
    detokenized_texts = [bpe.detokenize(token_ids) for token_ids in token_id_sequences]

    assert detokenized_texts == expected_texts
