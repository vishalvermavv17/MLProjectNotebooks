#!/usr/bin/env python
# coding: utf-8

# # Locality Sensitive Hashing (LSH) - Cosine Distance

# Locality Sensitive Hashing (LSH) provides for a fast, efficient approximate nearest neighbor search. The algorithm scales well with respect to the number of data points as well as dimensions.
# 
# In this notebook, we will
# * Implement the LSH algorithm for approximate nearest neighbor search
# * Examine the accuracy for different documents by comparing against brute force search, and also contrast runtimes
# * Explore the role of the algorithm’s tuning parameters in the accuracy of the method

# ## Import necessary packages

# In[128]:


from __future__ import print_function # to conform python 2.x print to python 3.x
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
import time
from copy import copy
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load in the Wikipedia dataset

# In[129]:


df = pd.read_csv('../people_wiki_csv/people_wiki.csv')


# Taking subset of given dataset as corpus creation of TF-IDF is time consuming. 

# In[130]:


df = df.sample(n=20000)
df.reset_index(inplace=True, drop=True)


# For this assignment, let us assign a unique ID to each document.

# In[131]:


df['id'] = df.index


# In[132]:


df.info()
df.head()


# Creating word cospus of dataset using TF-IDF

# In[133]:


from sklearn.feature_extraction.text import TfidfVectorizer


tfidf = TfidfVectorizer(
    analyzer='word',
    min_df=0,
    stop_words='english')
X_tfidf = tfidf.fit_transform(df['text'])
X_tfidf


# In[134]:


def get_similarity_items(X_tfidf, item_id, topn=5):
    """
    Get the top similar items for a given item id.
    The similarity measure here is based on cosine distance.
    """
    query = X_tfidf[item_id]
    scores = X_tfidf.dot(query.T).toarray().ravel()
    best_docs_idx = np.argpartition(scores, -topn)[-topn:]
    return sorted(zip(best_docs_idx, scores[best_docs_idx]), key=lambda x: -x[1])

similar_items = get_similarity_items(X_tfidf, item_id=1)

# an item is always most similar to itself, in real-world
# scenario we might want to filter itself out from the output
for similar_item, similarity in similar_items:
    item_description = df.loc[similar_item, 'text']
    print('similar item id: ', similar_item)
    print('cosine similarity: ', similarity)
    print('item description: ', item_description)
    print()
    


# ## Getting Started with LSH

# In[135]:


def generate_random_vectors(dim, n_vectors):
    """
    generate random projection vectors
    the dims comes first in the matrix's shape,
    so we can use it for matrix multiplication.
    """
    return np.random.randn(dim, n_vectors)


# In[136]:


vocab_size = len(tfidf.get_feature_names_out())
print('vocabulary size: ', vocab_size)

np.random.seed(0)
n_vectors = 16
random_vectors = generate_random_vectors(vocab_size, n_vectors)
print('dimension: ', random_vectors.shape)
random_vectors


# Next, we'd like to decide which bin each documents should go. Since 16 random vectors were generated in the previous cell, we have 16 bits to represent the bin index. The first bit is given by the sign of the dot product between the first random vector and the document's TF-IDF vector, and so on.

# In[137]:


# use one data point's tfidf representation as an example
data_point = X_tfidf[0]

# True if positive sign; False if negative sign
bin_indices_bits = data_point.dot(random_vectors) >= 0
print('dimension: ', bin_indices_bits.shape)
bin_indices_bits


# All documents that obtain exactly this vector will be assigned to the same bin. One further preprocessing step we'll perform is to convert this resulting bin into integer representation. This makes it convenient for us to refer to individual bins.

# In[138]:


bin_indices_bits = data_point.dot(random_vectors) >= 0

# https://wiki.python.org/moin/BitwiseOperators
# x << y is the same as multiplying x by 2 ** y
powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
print(powers_of_two)

# final integer representation of individual bins
bin_indices = bin_indices_bits.dot(powers_of_two)
print(bin_indices)


# We can repeat the identical operation on all documents in our dataset and compute the corresponding bin. We'll again leverage matrix operations so that no explicit loop is needed.

# In[139]:


# we can do it for the entire corpus
bin_indices_bits = X_tfidf.dot(random_vectors) >= 0
print(bin_indices_bits.shape)
bin_indices = bin_indices_bits.dot(powers_of_two)
bin_indices.shape


# bin_indices represent the bin index number for all documents in corpus.

# Now we are ready to complete the following function. Given the integer bin indices for the documents, we would curate the list of document IDs that belong to each bin. Since a list is to be maintained for each unique bin index, a dictionary of lists is used.

# In[140]:


from collections import defaultdict

def train_lsh(X_tfidf, n_vectors, seed=None):    
    if seed is not None:
        np.random.seed(seed)
        
    dim = X_tfidf.shape[1]
    random_vectors = generate_random_vectors(dim, n_vectors)
    
    # partition data points into bins,
    # and encode bin index bits into integers
    bin_indices_bits = X_tfidf.dot(random_vectors) >= 0
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
    bin_indices = bin_indices_bits.dot(powers_of_two)
    
    # update `table` so that `table[i]` is the list of document ids with bin index equal to i
    table = defaultdict(list)
    
    for idx, bin_index in enumerate(bin_indices):
        table[bin_index].append(idx)
        
    # note that we're storing the bin_indices here
    # so we can do some ad-hoc checking with it,
    # this isn't actually required
    model = {'table': table,
             'random_vectors': random_vectors,
             'bin_indices': bin_indices,
             'bin_indices_bits': bin_indices_bits}
    return model


# train the model
n_vectors = 16
model = train_lsh(X_tfidf, n_vectors, seed=0)


# After generating our LSH model, let's examine the generated bins to get a deeper understanding of them. Recall that during the background section, given a product's tfidf vector representation, we were able to find its similar product using standard cosine similarity. Here, we will look at these similar products' bins to see if the result matches intuition. Remember the idea behind LSH is that similar data points will tend to fall into nearby bins.

# In[141]:


# comparison
similar_item_ids = [similar_item for similar_item, _ in similar_items]
bits1 = model['bin_indices_bits'][similar_item_ids[0]]
bits2 = model['bin_indices_bits'][similar_item_ids[1]]

print('bits 1: ', bits1)
print('bits 2: ', bits2)
print('Number of agreed bins: ', np.sum(bits1 == bits2))


# Looking at the result above, it does seem like LSH is doing what its intended to do: i.e., similar data points will agree upon more bin indices than dissimilar data points, however, in a high-dimensional space such as text features, we can get unlucky with our selection of only a few random vectors such that dissimilar data points go into the same bin while similar data points fall into different bins. Hence, given a query document, we should consider all documents in the nearby bins and sort them according to their actual distances from the query.

# ## Querying the LSH model

# Let us first implement the logic for searching nearby neighbors, which goes like this:
# ```
# 1. Let L be the bit representation of the bin that contains the query documents.
# 2. Consider all documents in bin L.
# 3. Consider documents in the bins whose bit representation differs from L by 1 bit.
# 4. Consider documents in the bins whose bit representation differs from L by 2 bits.
# ...
# ```

# To obtain candidate bins that differ from the query bin by some number of bits, we use `itertools.combinations`, which produces all possible subsets of a given list. See [this documentation](https://docs.python.org/3/library/itertools.html#itertools.combinations) for details.
# ```
# 1. Decide on the search radius r. This will determine the number of different bits between the two vectors.
# 2. For each subset (n_1, n_2, ..., n_r) of the list [0, 1, 2, ..., num_vector-1], do the following:
#    * Flip the bits (n_1, n_2, ..., n_r) of the query bin to produce a new bit vector.
#    * Fetch the list of documents belonging to the bin indexed by the new bit vector.
#    * Add those documents to the candidate set.
# ```
# 
# Each line of output from the following cell is a 3-tuple indicating where the candidate bin would differ from the query bin. For instance,
# ```
# (0, 1, 3)
# ```
# indicates that the candiate bin differs from the query bin in first, second, and fourth bits.

# In[142]:


from itertools import combinations


def search_nearby_bins(query_bin_bits, table, search_radius=3, candidate_set=None):
    """
    For a given query vector and trained LSH model's table
    return all candidate neighbors with the specified search radius.
    
    Example
    -------
    model = train_lsh(X_tfidf, n_vectors=16, seed=0)
    query = model['bin_index_bits'][0]  # vector for the first document
    candidates = search_nearby_bins(query, model['table'])
    """
    if candidate_set is None:
        candidate_set = set()

    n_vectors = query_bin_bits.shape[0]
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
    
    for different_bits in combinations(range(n_vectors), search_radius):
        bits_index = list(different_bits)
        alternate_bits = query_bin_bits.copy()
        alternate_bits[bits_index] = np.logical_not(alternate_bits[bits_index])
        
        # convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)
        
        # fetch the list of documents belonging to
        # the bin indexed by the new bit vector,
        # then add those documents to candidate_set;
        # make sure that the bin exists in the table
        if nearby_bin in table:
            candidate_set.update(table[nearby_bin])
            
    return candidate_set


# The next code chunk uses this searching for nearby bins logic to search for similar documents and return a dataframe that contains the most similar data points according to LSH and their corresponding distances.

# In[143]:


from sklearn.metrics.pairwise import pairwise_distances


def get_nearest_neighbors(X_tfidf, query_vector, model, max_search_radius=3):
    table = model['table']
    random_vectors = model['random_vectors']
    
    # compute bin index for the query vector, in bit representation.
    bin_index_bits = np.ravel(query_vector.dot(random_vectors) >= 0)
    
    # search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in range(max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, candidate_set)
        
    # sort candidates by their true distances from the query
    candidate_list = list(candidate_set)
    candidates = X_tfidf[candidate_list]
    distance = pairwise_distances(candidates, query_vector, metric='cosine').flatten()
    
    distance_col = 'distance'
    nearest_neighbors = pd.DataFrame({
        'id': candidate_list, distance_col: distance
    }).sort_values(distance_col).reset_index(drop=True)
    
    return nearest_neighbors
    
    


# In[144]:


print('original similar items:\n' + str(similar_items))

item_id = 1
query_vector = X_tfidf[item_id]
nearest_neighbors = get_nearest_neighbors(X_tfidf, query_vector, model, max_search_radius=5)
print('dimension: ', nearest_neighbors.shape)
nearest_neighbors.head()


# We can observe from the result above that when we use a max_search_radius of 5, our LSH-based similarity search wasn't capable of retrieving the actual most similar items to our target data point, this is expected as we have mentioned LSH is an approximate nearest neighborhood search method. However, if we were to increase the radius parameter to 12, we can in fact retrieve all the actual most similar items.

# In[145]:


nearest_neighbors = get_nearest_neighbors(X_tfidf, query_vector, model, max_search_radius=12)
print('dimension: ', nearest_neighbors.shape)
nearest_neighbors.head()


# In this article, we saw that LSH performs an efficient neighbor search by randomly partitioning all reference data points into different bins, when it comes to the similarity search stage, it will only consider searching within data points that fall within the same bin. Then a radius parameter gives the end-user full control over the trade-off between the speed of the search versus the quality of the nearest neighbors.

# ## REFERENCE

# [Jupyter Notebook: Locality Sensitive Hashing](http://ethen8181.github.io/machine-learning/recsys/content_based/lsh_text.html#Getting-Started-with-LSH)
