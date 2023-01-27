from collections import defaultdict
from typing import Callable, Iterator, List, Optional
from cleantext import clean

from doccano_transformer import utils


class Example:
    def is_valid(self, raise_exception: Optional[bool] = True) -> None:
        raise NotImplementedError


class NERExample:
    def __init__(self, raw: dict) -> None:
        self.raw = raw
        self.id = raw["id"]
        self.text = raw["text"]
        self.sentences = utils.split_sentences(raw["text"])
        self.sentences = [x for x in self.sentences if len(x.strip()) > 0]
        self.sentence_offsets = utils.get_offsets(raw["text"], self.sentences)
        self.sentence_offsets.append(len(raw["text"]))

    @property
    def labels(self):
        if "annotations" in self.raw:
            labels = defaultdict(list)
            for annotation in self.raw["annotations"]:
                labels["DEFAULT"].append(
                    [
                        annotation["start_offset"],
                        annotation["end_offset"],
                        annotation["label"],
                    ]
                )
            return labels
        elif "label" in self.raw:
            labels = defaultdict(list)
            for label in self.raw["label"]:
                # TODO: This format doesn't have a user field currently.
                # So this method uses the user 0 for all label.
                labels[0].append(label)
            return labels
        else:
            raise KeyError('The file should includes either "labels" or "annotations".')

    def get_tokens_and_token_offsets(self, tokenizer):
        tokens = [tokenizer(sentence) for sentence in self.sentences]
        token_offsets = [
            utils.get_offsets(sentence, tokens, offset)
            for sentence, tokens, offset in zip(
                self.sentences, tokens, self.sentence_offsets
            )
        ]
        return tokens, token_offsets

    def is_valid(self, raise_exception: Optional[bool] = True) -> bool:
        return True

    def to_conll2003(
        self, tokenizer: Callable[[str], List[str]], selected_label: str
    ) -> Iterator[dict]:
        all_tokens, all_token_offsets = self.get_tokens_and_token_offsets(tokenizer)
        #       import pdb

        #        pdb.set_trace()

        # if len(self.labels) == 0:
        #    self.labels[0] = []
        for user, labels in self.labels.items():
      

            if selected_label is not None:
            #    print("FILTERING")
                labels = [x for x in labels if x[2] == selected_label]
           # print([self.raw['text'][x[0]:x[1]]  for x in labels])
            ok_labels = set([self.raw['text'][x[0]:x[1]]  for x in labels])
            label_split = [[] for _ in range(len(self.sentences))]
            for label in labels:
                for i, (start, end) in enumerate(
                    zip(self.sentence_offsets, self.sentence_offsets[1:])
                ):
                    if start <= label[0] <= label[1] <= end:
                        label_split[i].append(label)
            # lines = ["-DOCSTART- -X- -X- O\n\n"]
            lines = []
            tot_tokens = 0
            for tokens, offsets, label in zip(
                all_tokens, all_token_offsets, label_split
            ):
              #  if tokens[0] == "Applicazioni":
             #       import pdb; pdb.set_trace()
                tags = utils.create_bio_tags(tokens, offsets, label)
                print(tokens, tags)
                tot_tokens += len(tokens)
                if tot_tokens > 50:
                    SPLIT = True
                    tot_tokens = 0
                else:
                    SPLIT = False
                for token, tag in zip(tokens, tags):
                    token = clean(
                        token,
                        no_emoji=True,
                        fix_unicode=True,
                        lower=False,  # Convert to lowercase
                    )  # Remove all punctuations)
                    lines.append(f"{token} _ _ {tag}\n")

                if SPLIT:
                    lines.append("\n")
            yield {"user": user, "data": "".join(lines)}

    def to_spacy(self, tokenizer: Callable[[str], List[str]]) -> Iterator[dict]:
        all_tokens, all_token_offsets = self.get_tokens_and_token_offsets(tokenizer)
        for user, labels in self.labels.items():
            label_split = [[] for _ in range(len(self.sentences))]
            for label in labels:
                for i, (start, end) in enumerate(
                    zip(self.sentence_offsets, self.sentence_offsets[1:])
                ):
                    if start <= label[0] <= label[1] <= end:
                        label_split[i].append(label)

            data = {"raw": self.text}
            sentences = []
            for tokens, offsets, label in zip(
                all_tokens, all_token_offsets, label_split
            ):
                tokens = utils.convert_tokens_and_offsets_to_spacy_tokens(
                    tokens, offsets
                )
                tags = biluo_tags_from_offsets(tokens, label)
                tokens_for_spacy = []
                for i, (token, tag, offset) in enumerate(zip(tokens, tags, offsets)):
                    tokens_for_spacy.append({"id": i, "orth": str(token), "ner": tag})
                sentences.append({"tokens": tokens_for_spacy})
            data["sentences"] = sentences
            yield {"user": user, "data": {"id": self.id, "paragraphs": [data]}}
