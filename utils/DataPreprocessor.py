import numpy as np
import glob
import os
import sys
from collections import defaultdict as dd
import cPickle
from config import MAX_WORD_LEN
from local_config import *
import gc

from nltk.tag.stanford import StanfordNERTagger, StanfordPOSTagger
ner_tagger = StanfordNERTagger(NER_MODEL_PATH, NER_JAR_PATH, java_options='')
pos_tagger = StanfordPOSTagger(POS_MODEL_PATH, POS_JAR_PATH, java_options='')

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

SYMB_BEGIN = "@begin"
SYMB_END = "@end"
BIN_NUM = 5

D_NER_IDX, D_POS_IDX, Q_NER_IDX, Q_POS_IDX = 8, 9, 10, 11

lem_cache = {}

def lemmatize(word_):
    if word_ in lem_cache:
        return lem_cache[word_]

    word = word_.decode('utf-8', 'ignore')
    word_n = wordnet_lemmatizer.lemmatize(word)
    if word_n != word:
        lem_cache[word_] = word_n
    else:
        lem_cache[word_] = wordnet_lemmatizer.lemmatize(word, pos = 'v')
    return lem_cache[word_]

def get_tag(tagged):
    return [tag[1] for tag in tagged]

class Data:

    def __init__(self, dictionary, num_entities, training, validation, test):
        self.dictionary = dictionary
        self.training = training
        self.validation = validation
        self.test = test
        self.vocab_size = len(dictionary[0])
        self.num_chars = len(dictionary[1])
        self.num_entities = num_entities
        self.inv_dictionary = {v:k for k,v in dictionary[0].items()}

class DataPreprocessor:

    def preprocess(self, question_dir, use_chars=True):
        """
        preprocess all data into a standalone Data object.
        the training set will be left out (to save debugging time) when no_training_set is True.
        """
        vocab_f = os.path.join(question_dir,"vocab.txt")
        freq_f = os.path.join(question_dir, "freq.txt")
        word_dictionary, char_dictionary, num_entities, word2freq_old = \
                self.make_dictionary(question_dir, vocab_file=vocab_f, freq_file = freq_f)

        word2freq = {}
        for k, v in word2freq_old.iteritems():
            word2freq[word_dictionary[k]] = v

        dictionary = (word_dictionary, char_dictionary, word2freq)

        print "preparing training data ..."
        # training = self.parse_all_files(question_dir + "/training", dictionary, use_chars, question_dir + '/dev_cache.pkl')
        training = self.parse_all_files(question_dir + "/training", dictionary, use_chars, question_dir + '/train_cache.pkl')
        print "preparing validation data ..."
        validation = self.parse_all_files(question_dir + "/validation", dictionary, use_chars, question_dir + '/dev_cache.pkl')
        print "preparing test data ..."
        test = self.parse_all_files(question_dir + "/test", dictionary, use_chars, question_dir + '/test_cache.pkl')

        feat2index, feat_cnt = {}, 0
        for q in training + validation + test:
            if q[D_NER_IDX] is not None:
                for tag in q[D_NER_IDX]:
                    if tag not in feat2index:
                        feat2index[tag] = feat_cnt
                        feat_cnt += 1
            if q[Q_NER_IDX] is not None:
                for tag in q[Q_NER_IDX]:
                    if tag not in feat2index:
                        feat2index[tag] = feat_cnt
                        feat_cnt += 1
            if q[D_POS_IDX] is not None:
                for tag in q[D_POS_IDX]:
                    if tag not in feat2index:
                        feat2index[tag] = feat_cnt
                        feat_cnt += 1
            if q[Q_POS_IDX] is not None:
                for tag in q[Q_POS_IDX]:
                    if tag not in feat2index:
                        feat2index[tag] = feat_cnt
                        feat_cnt += 1
        for i in range(BIN_NUM):
            feat2index['FREQ-{}'.format(i)] = feat_cnt
            feat_cnt += 1
        print 'use_char_fdim', feat_cnt
        dictionary = dictionary + (feat2index, feat_cnt)

        data = Data(dictionary, num_entities, training, validation, test)
        return data

    def make_dictionary(self, question_dir, vocab_file, freq_file):

        if os.path.exists(vocab_file) and os.path.exists(freq_file):
            print "loading vocabularies from " + vocab_file + " ..."
            vocabularies = map(lambda x:x.strip(), open(vocab_file).readlines())
            word2freq = cPickle.load(open(freq_file))
        else:
            print "no " + vocab_file + " found, constructing the vocabulary list ..."

            fnames = []
            fnames += glob.glob(question_dir + "/test/*.question")
            fnames += glob.glob(question_dir + "/validation/*.question")
            fnames += glob.glob(question_dir + "/training/*.question")

            vocab_set = set()
            n = 0.
            word2freq = dd(int)
            for fname in fnames:
                
                fp = open(fname)
                fp.readline()
                fp.readline()
                document = fp.readline().split()
                fp.readline()
                query = fp.readline().split()
                fp.close()

                vocab_set |= set(document) | set(query)

                for word in document:
                    word2freq[word] += 1
                for word in query:
                    word2freq[word] += 1
                word2freq[SYMB_BEGIN] += 2
                word2freq[SYMB_END] += 2

                # show progress
                n += 1
                if n % 10000 == 0:
                    print '%3d%%' % int(100*n/len(fnames))

            entities = set(e for e in vocab_set if e.startswith('@entity'))

            # @placehoder, @begin and @end are included in the vocabulary list
            tokens = vocab_set.difference(entities)
            tokens.add(SYMB_BEGIN)
            tokens.add(SYMB_END)

            vocabularies = list(entities)+list(tokens)

            print "writing vocabularies to " + vocab_file + " ..."
            vocab_fp = open(vocab_file, "w")
            vocab_fp.write('\n'.join(vocabularies))
            vocab_fp.close()

            freqs = [v for k, v in word2freq.iteritems()]
            freqs.sort()
            freq2index = {}
            bin_size = len(freqs) / BIN_NUM + 1
            for i, start in enumerate(range(0, len(freqs), bin_size)):
                end = min(start + bin_size, len(freqs))
                for j in range(start, end):
                    freq2index[freqs[j]] = i
            for k in word2freq.keys():
                word2freq[k] = freq2index[word2freq[k]]
            cPickle.dump(word2freq, open(freq_file, 'w'), cPickle.HIGHEST_PROTOCOL)

        vocab_size = len(vocabularies)
        word_dictionary = dict(zip(vocabularies, range(vocab_size)))
        char_set = set([c for w in vocabularies for c in list(w)])
        char_set.add(' ')
        char_dictionary = dict(zip(list(char_set), range(len(char_set))))
        num_entities = len([v for v in vocabularies if v.startswith('@entity')])
        print "vocab_size = %d" % vocab_size
        print "num characters = %d" % len(char_set)
        print "%d anonymoused entities" % num_entities
        print "%d other tokens (including @placeholder, %s and %s)" % (
                vocab_size-num_entities, SYMB_BEGIN, SYMB_END)

        return word_dictionary, char_dictionary, num_entities, word2freq

    def parse_one_file(self, fname, dictionary, use_chars):
        """
        parse a *.question file into tuple(document, query, answer, filename)
        """
        w_dict, c_dict = dictionary[0], dictionary[1]
        raw = open(fname).readlines()
        doc_raw = raw[2].split() # document
        qry_raw = raw[4].split() # query
        ans_raw = raw[6].strip() # answer
        cand_raw = map(lambda x:x.strip().split(':')[0].split(), 
                raw[8:]) # candidate answers

        # wrap the query with special symbols
        qry_raw.insert(0, SYMB_BEGIN)
        qry_raw.append(SYMB_END)
        try:
            cloze = qry_raw.index('@placeholder')
        except ValueError:
            print '@placeholder not found in ', fname, '. Fixing...'
            at = qry_raw.index('@')
            qry_raw = qry_raw[:at] + [''.join(qry_raw[at:at+2])] + qry_raw[at+2:]
            cloze = qry_raw.index('@placeholder')

        match = np.zeros((len(doc_raw), len(qry_raw)), dtype = np.int32)
        for i, word_i in enumerate(doc_raw):
            for j, word_j in enumerate(qry_raw):
                if word_i == word_j or lemmatize(word_i) == lemmatize(word_j):
                    match[i, j] = 1

        # tokens/entities --> indexes
        doc_words = map(lambda w:w_dict[w], doc_raw)
        qry_words = map(lambda w:w_dict[w], qry_raw)
        if use_chars:
            doc_chars = map(lambda w:map(lambda c:c_dict.get(c,c_dict[' ']), 
                list(w)[:MAX_WORD_LEN]), doc_raw)
            qry_chars = map(lambda w:map(lambda c:c_dict.get(c,c_dict[' ']), 
                list(w)[:MAX_WORD_LEN]), qry_raw)
        else:
            doc_chars, qry_chars = [], []
        ans = map(lambda w:w_dict.get(w,0), ans_raw.split())
        cand = [map(lambda w:w_dict.get(w,0), c) for c in cand_raw]

        d_ner, d_pos, q_ner, q_pos = doc_raw, doc_raw, qry_raw, qry_raw

        return doc_words, qry_words, ans, cand, doc_chars, qry_chars, cloze, match, d_ner, d_pos, q_ner, q_pos

    def parse_all_files(self, directory, dictionary, use_chars, cache_file):
        """
        parse all files under the given directory into a list of questions,
        where each element is in the form of (document, query, answer, filename)
        """
        if os.path.exists(cache_file):
            gc.disable()
            temp = cPickle.load(open(cache_file))
            gc.enable()
            return temp
        all_files = glob.glob(directory + '/*.question')
        questions = []
        for i, f in enumerate(all_files):
            if i % 10000 == 0:
                print 'parsing {}'.format(i)
            questions.append(self.parse_one_file(f, dictionary, use_chars) + (f,))
        questions = self.parse_ner_pos(questions)
        cPickle.dump(questions, open(cache_file, 'w'), cPickle.HIGHEST_PROTOCOL)
        return questions

    def parse_ner_pos(self, questions):
        questions = [list(q) for q in questions]

        BATCH_SIZE = 10000
        for start in range(0, len(questions), BATCH_SIZE):
            end = min(len(questions), start + BATCH_SIZE)
            print 'ner pos {} {}'.format(start, end)
            d_ner_sents = ner_tagger.tag_sents([q[D_NER_IDX] for q in questions[start: end]])
            q_ner_sents = ner_tagger.tag_sents([q[Q_NER_IDX] for q in questions[start: end]])
            d_pos_sents = pos_tagger.tag_sents([q[D_POS_IDX] for q in questions[start: end]])
            q_pos_sents = pos_tagger.tag_sents([q[Q_POS_IDX] for q in questions[start: end]])
            for i, q in enumerate(questions[start: end]):
                q[D_NER_IDX] = get_tag(d_ner_sents[i]) if len(q[D_NER_IDX]) == len(d_ner_sents[i]) else None
                q[Q_NER_IDX] = get_tag(q_ner_sents[i]) if len(q[Q_NER_IDX]) == len(q_ner_sents[i]) else None
                q[D_POS_IDX] = get_tag(d_pos_sents[i]) if len(q[D_POS_IDX]) == len(d_pos_sents[i]) else None
                q[Q_POS_IDX] = get_tag(q_pos_sents[i]) if len(q[Q_POS_IDX]) == len(q_pos_sents[i]) else None
        questions = [tuple(q) for q in questions]
        return questions

    def gen_text_for_word2vec(self, question_dir, text_file):

            fnames = []
            fnames += glob.glob(question_dir + "/training/*.question")

            out = open(text_file, "w")

            for fname in fnames:
                
                fp = open(fname)
                fp.readline()
                fp.readline()
                document = fp.readline()
                fp.readline()
                query = fp.readline()
                fp.close()
                
                out.write(document.strip())
                out.write(" ")
                out.write(query.strip())

            out.close()

if __name__ == '__main__':
    dp = DataPreprocessor()
    dp.gen_text_for_word2vec("cnn/questions", "/tmp/cnn_questions.txt")

