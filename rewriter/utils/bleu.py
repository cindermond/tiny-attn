import math
import re
from past.utils import old_div
from collections import defaultdict

class NGramScore(object):
    """Base class for BLEU & NIST, providing tokenization and some basic n-gram matching
    functions."""

    def __init__(self, max_ngram, case_sensitive):
        """Create the scoring object.
        @param max_ngram: the n-gram level to compute the score for
        @param case_sensitive: use case-sensitive matching?
        """
        self.max_ngram = max_ngram
        self.case_sensitive = case_sensitive

    def reset(self):
        """Reset the object, zero all counters."""
        raise NotImplementedError()

    def append(self, pred_sent, ref_sents):
        """Add a sentence to the statistics.
        @param pred_sent: system output / predicted sentence
        @param ref_sents: reference sentences
        """
        raise NotImplementedError()

    def score(self):
        """Compute the current score based on sentences added so far."""
        raise NotImplementedError()

    def ngrams(self, n, sent):
        """Given a sentence, return n-grams of nodes for the given N. Lowercases
        everything if the measure should not be case-sensitive.

        @param n: n-gram 'N' (1 for unigrams, 2 for bigrams etc.)
        @param sent: the sent in question
        @return: n-grams of nodes, as tuples of tuples (t-lemma & formeme)
        """
        if not self.case_sensitive:
            return list(zip(*[[tok.lower() for tok in sent[i:]] for i in range(n)]))
        return list(zip(*[sent[i:] for i in range(n)]))

    def check_tokenized(self, pred_sent, ref_sents):
        """Tokenize the predicted sentence and reference sentences, if they are not tokenized.
        @param pred_sent: system output / predicted sentence
        @param ref_sent: a list of corresponding reference sentences
        @return: a tuple of (pred_sent, ref_sent) where everything is tokenized
        """
        # tokenize if needed
        pred_sent = pred_sent if isinstance(pred_sent, list) else self.tokenize(pred_sent)
        ref_sents = [ref_sent if isinstance(ref_sent, list) else self.tokenize(ref_sent)
                     for ref_sent in ref_sents]
        return pred_sent, ref_sents

    def get_ngram_counts(self, n, sents):
        """Returns a dictionary with counts of all n-grams in the given sentences.
        @param n: the "n" in n-grams (how long the n-grams should be)
        @param sents: list of sentences for n-gram counting
        @return: a dictionary (ngram: count) listing counts of n-grams attested in any of the sentences
        """
        merged_ngrams = {}

        for sent in sents:
            ngrams = defaultdict(int)

            for ngram in self.ngrams(n, sent):
                ngrams[ngram] += 1
            for ngram, cnt in ngrams.items():
                merged_ngrams[ngram] = max((merged_ngrams.get(ngram, 0), cnt))
        return merged_ngrams

    def tokenize(self, sent):
        """This tries to mimic multi-bleu-detok from Moses, and by extension mteval-v13b.
        Code taken directly from there and attempted rewrite into Python."""
        # language-independent part:
        sent = re.sub(r'<skipped>', r'', sent)  # strip "skipped" tags
        sent = re.sub(r'-\n', r'', sent)  # strip end-of-line hyphenation and join lines
        sent = re.sub(r'\n', r' ', sent)  # join lines
        sent = re.sub(r'&quot;', r'"', sent)  # convert SGML tag for quote to "
        sent = re.sub(r'&amp;', r'&', sent)  # convert SGML tag for ampersand to &
        sent = re.sub(r'&lt;', r'<', sent)  # convert SGML tag for less-than to >
        sent = re.sub(r'&gt;', r'>', sent)  # convert SGML tag for greater-than to <

        # language-dependent part (assuming Western languages):
        sent = " " + sent + " "  # pad with spaces
        sent = re.sub(r'([\{-\~\[-\` -\&\(-\+\:-\@\/])', r' \1 ', sent)  # tokenize punctuation
        sent = re.sub(r'([^0-9])([\.,])', r'\1 \2 ', sent)  # tokenize period and comma unless preceded by a digit
        sent = re.sub(r'([\.,])([^0-9])', r' \1 \2', sent)  # tokenize period and comma unless followed by a digit
        sent = re.sub(r'([0-9])(-)', r'\1 \2 ', sent)  # tokenize dash when preceded by a digit
        sent = re.sub(r'\s+', r' ', sent)  # one space only between words
        sent = sent.strip()  # remove padding

        return sent.split(' ')



class BLEUScore(NGramScore):
    """An accumulator object capable of computing BLEU score using multiple references.

    The BLEU score is always smoothed a bit so that it's never undefined. For sentence-level
    measurements, proper smoothing should be used via the smoothing parameter (set to 1.0 for
    the same behavior as default Moses's MERT sentence BLEU).
    """

    TINY = 1e-15
    SMALL = 1e-9

    def __init__(self, max_ngram=4, case_sensitive=False, smoothing=0.0):
        """Create the scoring object.
        @param max_ngram: the n-gram level to compute the score for (default: 4)
        @param case_sensitive: use case-sensitive matching (default: no)
        @param smoothing: constant to add for smoothing (defaults to 0.0, sentBLEU uses 1.0)
        """
        super(BLEUScore, self).__init__(max_ngram, case_sensitive)
        self.smoothing = smoothing
        self.reset()

    def reset(self):
        """Reset the object, zero all counters."""
        self.ref_len = 0
        self.cand_lens = [0] * self.max_ngram
        self.hits = [0] * self.max_ngram

    def append(self, pred_sent, ref_sents):
        """Append a sentence for measurements, increase counters.

        @param pred_sent: the system output sentence (string/list of tokens)
        @param ref_sents: the corresponding reference sentences (list of strings/lists of tokens)
        """
        pred_sent, ref_sents = self.check_tokenized(pred_sent, ref_sents)

        # compute n-gram matches
        for i in range(self.max_ngram):
            self.hits[i] += self.compute_hits(i + 1, pred_sent, ref_sents)
            self.cand_lens[i] += len(pred_sent) - i

        # take the reference that is closest in length to the candidate
        # (if there are two of the same distance, take the shorter one)
        closest_ref = min(ref_sents, key=lambda ref_sent: (abs(len(ref_sent) - len(pred_sent)), len(ref_sent)))
        self.ref_len += len(closest_ref)

    def score(self):
        """Return the current BLEU score, according to the accumulated counts."""
        return self.bleu()

    def compute_hits(self, n, pred_sent, ref_sents):
        """Compute clipped n-gram hits for the given sentences and the given N

        @param n: n-gram 'N' (1 for unigrams, 2 for bigrams etc.)
        @param pred_sent: the system output sentence (tree/tokens)
        @param ref_sents: the corresponding reference sentences (list/tuple of trees/tokens)
        """
        merged_ref_ngrams = self.get_ngram_counts(n, ref_sents)
        pred_ngrams = self.get_ngram_counts(n, [pred_sent])

        hits = 0
        for ngram, cnt in pred_ngrams.items():
            hits += min(merged_ref_ngrams.get(ngram, 0), cnt)

        return hits

    def bleu(self):
        """Return the current BLEU score, according to the accumulated counts."""
        # brevity penalty (smoothed a bit: if candidate length is 0, we change it to 1e-5
        # to avoid division by zero)
        bp = 1.0
        if (self.cand_lens[0] <= self.ref_len):
            bp = math.exp(1.0 - old_div(self.ref_len,
                          (float(self.cand_lens[0]) if self.cand_lens[0] else 1e-5)))

        return bp * self.ngram_precision()

    def ngram_precision(self):
        """Return the current n-gram precision (harmonic mean of n-gram precisions up to max_ngram)
        according to the accumulated counts."""
        prec_log_sum = 0.0
        for n_hits, n_len in zip(self.hits, self.cand_lens):
            n_hits += self.smoothing  # pre-set smoothing
            n_len += self.smoothing
            n_hits = max(n_hits, self.TINY)  # forced smoothing just a litle to make BLEU defined
            n_len = max(n_len, self.SMALL)   # only applied for zeros
            prec_log_sum += math.log(old_div(n_hits, n_len))

        return math.exp((1.0 / self.max_ngram) * prec_log_sum)