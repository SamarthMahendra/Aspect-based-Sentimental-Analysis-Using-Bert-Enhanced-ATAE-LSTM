# Vocabulary Builder Class
from typing import List
from collections import Counter
import logging

class VocabularyBuilder:
    def __init__(self, min_freq: int = 2):
        self.word2idx = {'<pad>': 0, '<unk>': 1}
        self.idx2word = {0: '<pad>', 1: '<unk>'}
        self.word_freq = Counter()
        self.min_freq = min_freq

    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from list of texts"""
        for text in texts:
            words = text.split()
            self.word_freq.update(words)

        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1


    def text_to_indices(self, text: str) -> List[int]:
        """Convert text to list of indices"""
        return [self.word2idx.get(word, self.word2idx['<unk>']) for word in text.split()]