import json
import tempfile

from src.tokenizers.bpe import BPE, BPEConfig
from src.tokenizers.utils import InvertibleDict, InvertibleDictEncoder


class TestBPE:
    cfg = BPEConfig(base_vocab="bghnpsu", vocab_size=16)
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
            "un": 13,
            "unĠ": 14,
            "hug": 15,
        }
    )

    # TODO: implement later
    def test_normalize(self):
        example = "hello"
        expected = "hello"

        normalized = self.bpe.normalize(example)
        assert normalized == expected

    def test_pre_tokenize(self):
        example = "hug bug   pun"
        expected = "hugĠ bugĠ pun"

        pretoken_exp = self.bpe.pre_tokenize(example)
        assert pretoken_exp == expected

    def test_bpe_train(self):
        learned_vocab = self.bpe.get_vocab

        assert learned_vocab.keys() == self.expected_vocab.keys()

    def test_break_into_subwords(self):
        # Considers space as a character of the word
        texts = ["bug", "mug", "bug mug hug hugs!", "nun ", "nunĠets"]
        expected_tokens = [
            ["b", "ug"],
            ["m", "ug"],
            ["b", "ug", " ", "m", "ug", " ", "hug", " ", "hug", "s", "!"],
            ["n", "un", " "],
            ["n", "unĠ", "e", "t", "s"],
        ]

        tokens = [self.bpe._break_into_subwords(text) for text in texts]

        assert tokens == expected_tokens

    def test_tokenize(self):
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

        token_ids = [self.bpe.tokenize(text) for text in texts]
        expected_token_ids = [
            [self.bpe.vocab[t] for t in tokens] for tokens in expected_tokens
        ]

        assert token_ids == expected_token_ids

    def test_tokenize_batch(self):
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

        token_ids = self.bpe.tokenize_batch(texts)
        expected_token_ids = [
            [self.bpe.vocab[t] for t in tokens] for tokens in expected_tokens
        ]

        assert token_ids == expected_token_ids

    def test_detokenize(self):
        token_id_sequences = [
            [2, 5, 12, 3],
            [2, 0, 12, 3],
            [2, 5, 12, 4, 0, 12, 4, 15, 4, 15, 10, 0, 3],
        ]
        expected_texts = ["bug", "<unk>ug", "bug <unk>ug hug hugs<unk>"]

        detokenized_texts = [
            self.bpe.detokenize(token_ids) for token_ids in token_id_sequences
        ]

        assert detokenized_texts == expected_texts

    def test_detokenize_batch(self):
        token_id_sequences = [
            [2, 5, 12, 3],
            [2, 0, 12, 3],
            [2, 5, 12, 4, 0, 12, 4, 15, 4, 15, 10, 0, 3],
        ]
        expected_texts = ["bug", "<unk>ug", "bug <unk>ug hug hugs<unk>"]

        detokenized_texts = self.bpe.detokenize_batch(token_id_sequences)

        assert detokenized_texts == expected_texts

    def test_save(self):
        # use a temp file to avoid writing to disk
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            self.bpe.save(temp_file.name, overwrite=True)

            temp_file.seek(0)
            saved_vocab = json.load(temp_file)
            assert saved_vocab == self.bpe.get_vocab

    def test_load(self):
        # use a temp file to avoid writing to disk
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            json.dump(self.expected_vocab, temp_file, cls=InvertibleDictEncoder)
            temp_file_name = temp_file.name

        self.bpe.load(temp_file_name)
        assert self.bpe.get_vocab == self.expected_vocab
