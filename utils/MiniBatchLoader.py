import glob
import numpy as np
import random

import re
from config import MAX_WORD_LEN


class MiniBatchLoader():

    def __init__(self, questions, batch_size, dictionary, shuffle=True, sample=1.0, max_qry_len=None):
        self.batch_size = batch_size
        self.word2freq, self.feat2index, self.feat_cnt = dictionary[2], dictionary[3], dictionary[4]

        if sample==1.0: self.questions = questions
        else: self.questions = random.sample(questions, 
                int(sample*len(questions)))
        self.bins = self.build_bins(self.questions)
        if max_qry_len is None:
            self.max_qry_len = max(map(lambda x:len(x[1]), self.questions))
        else:
            self.max_qry_len = max_qry_len
        self.max_num_cand = max(map(lambda x:len(x[3]), self.questions))
        self.max_word_len = MAX_WORD_LEN
        self.shuffle = shuffle
        self.reset()

    def __iter__(self):
        """make the object iterable"""
        return self

    def build_bins(self, questions):
        """
        returns a dictionary
            key: document length (rounded to the powers of two)
            value: indexes of questions with document length equal to key
        """
        # round the input to the nearest power of two
        round_to_power = lambda x: 2**(int(np.log2(x-1))+1)

        doc_len = map(lambda x:round_to_power(len(x[0])), questions)
        bins = {}
        for i, l in enumerate(doc_len):
            if l not in bins:
                bins[l] = []
            bins[l].append(i)

        return bins

    def reset(self):
        """new iteration"""
        self.ptr = 0

        # randomly shuffle the question indices in each bin
        if self.shuffle:
            for ixs in self.bins.itervalues():
                random.shuffle(ixs)

        # construct a list of mini-batches where each batch is a list of question indices
        # questions within the same batch have identical max document length 
        self.batch_pool = []
        for l, ixs in self.bins.iteritems():
            n = len(ixs)
            batch_size = min(self.batch_size, 16 * 854 / l)
            k = n/batch_size if n % batch_size == 0 else n/batch_size+1
            ixs_list = [(ixs[batch_size*i:min(n, batch_size*(i+1))],l) for i in range(k)]
            self.batch_pool += ixs_list

        # randomly shuffle the mini-batches
        if self.shuffle:
            random.shuffle(self.batch_pool)

    def next(self):
        """load the next batch"""
        if self.ptr == len(self.batch_pool):
            self.reset()
            raise StopIteration()

        ixs = self.batch_pool[self.ptr][0]
        curr_max_doc_len = self.batch_pool[self.ptr][1]
        curr_batch_size = len(ixs)

        dw = np.zeros((curr_batch_size, curr_max_doc_len, 1), dtype='int32') # document words
        qw = np.zeros((curr_batch_size, self.max_qry_len, 1), dtype='int32') # query words
        c = np.zeros((curr_batch_size, curr_max_doc_len, self.max_num_cand), 
                dtype='int16')   # candidate answers
        cl = np.zeros((curr_batch_size,), dtype='int32') # position of cloze in query

        m_dw = np.zeros((curr_batch_size, curr_max_doc_len), dtype='int32')  # document word mask
        m_qw = np.zeros((curr_batch_size, self.max_qry_len), dtype='int32')  # query word mask
        m_c = np.zeros((curr_batch_size, curr_max_doc_len), dtype='int32') # candidate mask

        a = np.zeros((curr_batch_size, ), dtype='int32')    # correct answer
        fnames = ['']*curr_batch_size

        match_feat = np.zeros((curr_batch_size, curr_max_doc_len, self.max_qry_len), dtype = np.int32)
        use_char, use_char_q = np.zeros((curr_batch_size, curr_max_doc_len, self.feat_cnt), dtype = np.float32), np.zeros((curr_batch_size, self.max_qry_len, self.feat_cnt), dtype = np.float32)

        types = {}

        for n, ix in enumerate(ixs):

            doc_w, qry_w, ans, cand, doc_c, qry_c, cloze, match, d_ner, d_pos, q_ner, q_pos, fname = self.questions[ix]

            if match.shape[1] < self.max_qry_len:
                match_feat[n, :match.shape[0], :match.shape[1]] = match
            else:
                match_feat[n, :match.shape[0], :self.max_qry_len] = match[:, :self.max_qry_len]

            if d_ner is not None:
                for i, tag in enumerate(d_ner):
                    use_char[n, i, self.feat2index[tag]] = 1.0
            if d_pos is not None:
                for i, tag in enumerate(d_pos):
                    use_char[n, i, self.feat2index[tag]] = 1.0
            for i, word in enumerate(doc_w):
                use_char[n, i, self.feat2index['FREQ-{}'.format(self.word2freq[word])]] = 1.0

            if q_ner is not None:
                for i, tag in enumerate(q_ner):
                    if i >= self.max_qry_len: break
                    use_char_q[n, i, self.feat2index[tag]] = 1.0
            if q_pos is not None:
                for i, tag in enumerate(q_pos):
                    if i >= self.max_qry_len: break
                    use_char_q[n, i, self.feat2index[tag]] = 1.0
            for i, word in enumerate(qry_w):
                if i >= self.max_qry_len: break
                use_char_q[n, i, self.feat2index['FREQ-{}'.format(self.word2freq[word])]] = 1.0

            # document, query and candidates
            dw[n,:len(doc_w),0] = np.array(doc_w)
            if len(qry_w) < self.max_qry_len:
                qw[n,:len(qry_w),0] = np.array(qry_w)
            else:
                qw[n,:self.max_qry_len,0] = np.array(qry_w[:self.max_qry_len])
            m_dw[n,:len(doc_w)] = 1
            m_qw[n,:min(len(qry_w), self.max_qry_len)] = 1
            for it, word in enumerate(doc_c):
                wtuple = tuple(word)
                if wtuple not in types:
                    types[wtuple] = []
                types[wtuple].append((0,n,it))
            for it, word in enumerate(qry_c):
                if it >= self.max_qry_len: break
                wtuple = tuple(word)
                if wtuple not in types:
                    types[wtuple] = []
                types[wtuple].append((1,n,it))

            # search candidates in doc
            for it,cc in enumerate(cand):
                index = [ii for ii in range(len(doc_w)) if doc_w[ii] in cc]
                m_c[n,index] = 1
                c[n,index,it] = 1
                if ans==cc: a[n] = it # answer

            cl[n] = cloze if cloze < self.max_qry_len else self.max_qry_len - 1
            fnames[n] = fname

        # create type character matrix and indices for doc, qry
        dt = np.zeros((curr_batch_size, curr_max_doc_len), dtype='int32') # document token index
        qt = np.zeros((curr_batch_size, self.max_qry_len), dtype='int32') # query token index
        tt = np.zeros((len(types), self.max_word_len), dtype='int32') # type characters
        tm = np.zeros((len(types), self.max_word_len), dtype='int32') # type mask
        n = 0
        for k,v in types.iteritems():
            tt[n,:len(k)] = np.array(k)
            tm[n,:len(k)] = 1
            for (sw, bn, sn) in v:
                if sw==0: dt[bn,sn] = n
                else: qt[bn,sn] = n
            n += 1

        self.ptr += 1

        return dw, dt, qw, qt, a, m_dw, m_qw, tt, tm, c, m_c, cl, fnames, match_feat, use_char, use_char_q

def unit_test(mini_batch_loader):
    """unit test to validate MiniBatchLoader using max-frequency (exclusive).
    The accuracy should be around 0.37 and should be invariant over different batch sizes."""
    hits, n = 0., 0
    for d, q, a, m_d, m_q, c, m_c in mini_batch_loader:
        for i in xrange(len(d)):
            prediction, max_count = -1, 0
            for cand in c[i]:
                count = (d[i]==cand).sum() + (q[i]==cand).sum()
                if count > max_count and cand not in q[i]:
                    max_count = count
                    prediction = cand
            n += 1
            hits += a[i] == prediction
        acc = hits/n
        print acc

if __name__ == '__main__':

    from DataPreprocessor import *
    
    cnn = DataPreprocessor().preprocess("cnn/questions", no_training_set=True)
    mini_batch_loader = MiniBatchLoader(cnn.validation, 64)
    unit_test(mini_batch_loader)

