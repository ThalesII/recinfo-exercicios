#!/usr/bin/env python3

import numpy

documents = ['O peã e o caval são pec de xadrez. O caval é o melhor do jog.',
             'A jog envolv a torr, o peã e o rei.',
             'O peã lac o boi.',
             'Caval de rodei!',
             'Polic o jog no xadrez.']
stopwords = ['a', 'o', 'e', 'é', 'de', 'do', 'no', 'são']
separators = [' ', ',', '.', '!', '?']
query = 'jog xadrez'


class Corpus:
	def __init__(self, seps=' ', stop=[]):
		self._seps = seps
		self._stop = stop
		self._docs = []
		self._terms = []
		self._freqs = []
		self._ns = []

	def _preprocess(self, doc):
		doc = doc.lower()
		for sep in self._seps:
			doc = doc.replace(sep, ' ')
		doc = doc.split()
		return [t for t in doc if t not in self._stop]

	def _add_term(self, term):
		if term in self._terms:
			return
		self._terms.append(term)
		self._ns.append(0)
		for freq in self._freqs:
			freq.append(0)

	def _frequency(self, doc):
		return [doc.count(t) for t in self._terms]

	def _tf(self, freq):
		return [1 + numpy.log2(f) if f != 0 else 0 for f in freq]

	def _idf(self):
		N = len(self._docs)
		return [numpy.log2(N/n) for n in self._ns]

	def _weight(self, freq):
		tf = self._tf(freq)
		idf = self._idf()
		return [t*i for (t, i) in zip(tf, idf)]

	def _query_weight(self, freq):
		return self._weight(freq)

	def add_document(self, doc):
		self._docs.append(doc)
		doc = self._preprocess(doc)
		for term in doc:
			self._add_term(term)
		freq = self._frequency(doc)
		self._freqs.append(freq)
		for i, f in enumerate(freq):
			self._ns[i] += f > 0

	def query_weight(self, q):
		q = self._preprocess(q)
		freq = self._frequency(q)
		return self._query_weight(freq)

	def weight_matrix(self):
		return [self._weight(f) for f in self._freqs]


def main():
	corpus = Corpus(separators, stopwords)
	for doc in documents:
		corpus.add_document(doc)

	q = corpus.query_weight(query)
	m = corpus.weight_matrix()

	numpy.set_printoptions(precision=2, linewidth=90)
	print('query = ', repr(query))
	print(' w`=', numpy.array(q), end='\n\n')
	for i, doc in enumerate(documents):
		print('d%d =' % i, repr(doc))
		print(' w =', numpy.array(m[i]), end='\n\n')


if __name__ == '__main__':
	main()

