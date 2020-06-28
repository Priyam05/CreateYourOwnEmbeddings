# CreateYourOwnEmbeddings

Here, I have written code to approximate a Skip-gram word embeddings via positive pointwise mutual information (PPMI) and singular value decomposition (SVD), and evaluate your trained embeddings.

## Skip-gram Word Embedding as Implicit Matrix Factorization\
The goal of learning word embedding is to derive a low-dimensional continuous vector representation for words so that words that are syntactically or semantically related are closer in that vector space. One such method is Skip-gram model. Instead of training a Skip-gram word embeddings from scratch, we'll try to approximate one with PPMI and truncated SVD following the paper by Levy & Goldberg 1. In their paper, they show that Skip-gram word training method can be cast as weighted matrix factorization and that its objective
is implicitly factorizing a word-context matrix, whose cells are the pointwise mutual information (PMI) of the respective word and context pairs.

### Create Vocabulary
In this part, we will create vocabulary from the movie plot summaries 2 and use that as our corpus. Vocabulary, denoted as V is a dictionary of unique words present in our dataset or text.

### Build Term-Context Matrix
In this part, we will build a trem-context matrix form the plot summaries dataset. For a target word wi 2 V and window size L, we define the context window for wi as the words surrounding it in an L-sized window i.e. L words before and after that word in all the documents. If the defined window exceeds the document length, only consider the
the portion till the beginning/end of the document. Here, we only consider target words and context words that are in the vocabulary V. 

Instead of considering the entire document for obtaining the context window of a single word, it is more common to use a smaller context window. Like in section 6.3.2 in Jurafsky & Martin 3, we will calculate the term-context matrix M by calculating the number of each target-context pair (w; c) for all the words in the vocabulary. Here, target w is our word of interest and context c is a context word occurring near the target word w in the context window. The entry of the term-context matrix is defined as follow,

![Context Matrix details](/Images/ContextMatrix.PNG)

We use #(w; c) = count(c; ContextWindow(w;L)) to denote the number of times the context word c appears in the context window of the target word w
in the dataset. The collection of all the observed target-context pairs is denoted as D. 

The resulting term-context matrix is sparse; therefore, it is more efficient to store #(w; c), #(w), and #(c) separately than storing an entire matrix. In this part, we'll write code to calculate #(w; c), #(w), and #(c) that allow us to build our PPMI matrix in the next part.

### Build Positive Pointwise Mutual Information (PPMI) Matrix
Pointwise mutual information is an information-theoretic association measure between a pair of discrete outcomes x and y. In this case, PMI(w; c) measures the association between a target word w and a context word c by calculating the log of the ratio between their joint probability (the frequency in which they occur together) and their marginal probabilities (the frequency in which they occur independently). PMI can be estimated empirically by considering the actual number of observations in a corpus:

![PPMI](/Images/PPMI.PNG)

### Truncated SVD

In this part, we will obtain a dense low-dimensional vectors via truncated (rank-k) singular value decomposition - a basic algorithm from linear algebra which is used to achieve the optimal rank k factorization with respect to L2 loss. SVD factorizes the PPMI matrix into the product of 3 matrices U E V.T , where U and V are orthonormal and E is a diagonal matrix of singular values. Let Ek be the diagonal matrix formed from the top k singular values, and let Uk and Vk be the matrices produced by selecting the corresponding columns from U and V . Our PPMI matrix MPPMI is very sparse and hence can be approximated well via a rank-k singular value decomposition, MPPMI = Uk Ek Vk. T. We can use the resulting left singular vectors Uk of rank k as our vector representations for each of the words in our vocabulary.

### Evaluate Word Embeddings via Cosine Similarity

Using cosine similarity as a measure of distance (section 6.4 in Jurafsky & Martin 5), we can now find the closest words to a certain word in the word embeddings space via cosine similarity. We define cosine similarity as,
![CosineSimilarity](/Images/CS.PNG)

### Evaluate Word Embeddings via an Analogous Task

The embedding space is known to capture the semantic context of words.
![Evaluation](/Images/Evaluation.PNG)


