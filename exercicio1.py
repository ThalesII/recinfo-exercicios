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


def tokenize(text, seps):
	for sep in seps:
		text = text.replace(sep, ' ')
	return text.split()


def normalize(words, stop):
	return [w.lower() for w in words if w.lower() not in stop]


def gather_terms(docs):
	terms = set()
	for doc in docs: 
		terms = terms.union(doc)
	return list(terms)


def build_index(docs, terms):
	index = [[d.count(t) for d in docs] for t in terms]
	return numpy.array(index)


def query_and(query, index, terms):
	query = query.split()
	result = numpy.ones_like(index[0])
	for term in query:
		i = terms.index(term)
		result *= index[i]
	return result.astype(bool)


def query_or(query, index, terms):
	query = query.split()
	result = numpy.zeros_like(index[0])
	for term in query:
		i = terms.index(term)
		result += index[i]
	return result.astype(bool)


def main():
	docs = []
	for doc in documents:
		doc = tokenize(doc, separators)
		doc = normalize(doc, stopwords)
		docs.append(doc)
	terms = gather_terms(docs)
	index = build_index(docs, terms)
	result = query_and(query, index, terms)
	for i, r in enumerate(result):
		if r: print('d%d = "%s"' % (i+1, documents[i]))


if __name__ == '__main__':
	main()

