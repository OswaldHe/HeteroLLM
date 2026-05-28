import bm25s
import random
import time

L = 131072
D = 128
# Create your corpus here
corpus = []

for i in range(L):
    doc = ""
    for j in range(D):
        doc += str(random.randint(0, 65536)) + " "
    corpus.append(doc)

# Tokenize the corpus and only keep the ids (faster and saves memory)
corpus_tokens = bm25s.tokenize(corpus)

# Create the BM25 model and index the corpus
retriever = bm25s.BM25()
retriever.index(corpus_tokens)

# Query the corpus
query = ""

for i in range(64):
    query += str(random.randint(0, 65536)) + " "
query_tokens = bm25s.tokenize(query)

# Get top-k results as a tuple of (doc ids, scores). Both are arrays of shape (n_queries, k).
# To return docs instead of IDs, set the `corpus=corpus` parameter.
start = time.time()
results, scores = retriever.retrieve(query_tokens, k=64)
end = time.time()
print(f"Retrieved {len(results[0])} results in {end - start:.7f} seconds.")