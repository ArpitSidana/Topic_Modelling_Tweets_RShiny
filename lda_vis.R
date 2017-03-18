library("LDAvis")
library(NLP)
library(tm)
library("servr")
library(shiny)
library("lda")

# Load dataframe
tweets <- read.csv("A:/Rshiny_Github/Model_1_Dataset.csv")

stop_words <- stopwords("SMART")
tweet <- tweets$text

tweet <- tolower(tweet)  # lowercase
tweet <- gsub("'", "", tweet)  # remove apostrophes
tweet <- gsub("https", "", tweet)  # remove url
tweet <- gsub("[[:punct:]]", " ", tweet)  # replace punctuation with space
tweet <- gsub("[[:cntrl:]]", " ", tweet)  # replace control characters with space
tweet <- gsub("^[[:space:]]+", "", tweet) # remove whitespace at beginning of documents
tweet <- gsub("[[:space:]]+$", "", tweet) # remove whitespace at end of documents

# tokenize on space and output as a list:
doc.list <- strsplit(tweet, "[[:space:]]+")

# compute the table of terms:
term.table <- table(unlist(doc.list))
term.table <- sort(term.table, decreasing = TRUE)

# remove terms that are stop words or occur fewer than 10 times:
del <- names(term.table) %in% stop_words | term.table < 10
term.table <- term.table[!del]
vocab <- names(term.table)

# prepare documents for lda:
get.terms <- function(x) {
  index <- match(x, vocab)
  index <- index[!is.na(index)]
  rbind(as.integer(index - 1), as.integer(rep(1, length(index))))
}
documents <- lapply(doc.list, get.terms)

# Compute some statistics related to the data set:
D <- length(documents)  # number of documents (2,000)
W <- length(vocab)  # number of terms in the vocab (14,568)
doc.length <- sapply(documents, function(x) sum(x[2, ]))  # number of tokens per document 
N <- sum(doc.length)  # total number of tokens in the data 
term.frequency <- as.integer(term.table)  # frequencies of terms in the corpus 

# Fit the model:
library(lda)
set.seed(1)

fit <- lda.collapsed.gibbs.sampler(documents = documents, K = 10, vocab = vocab, 
                                   num.iterations = 250, alpha = 0.5, eta=0.5,
                                   initial = NULL, burnin = 0,
                                   compute.log.likelihood = TRUE)

#LDAvis
theta <- t(apply(fit$document_sums + 0.1, 2, function(x) x/sum(x)))
phi <- t(apply(t(fit$topics) + 0.1, 2, function(x) x/sum(x)))

tweetvis <- list(phi = phi,
                 theta = theta,
                 doc.length = doc.length,
                 vocab = vocab,
                 term.frequency = term.frequency)


# create visualization
json <- createJSON(phi = tweetvis$phi, 
                   theta = tweetvis$theta, 
                   doc.length = tweetvis$doc.length, 
                   vocab = tweetvis$vocab, 
                   term.frequency = tweetvis$term.frequency)

serVis(json, out.dir = tempfile(), open.browser = interactive())