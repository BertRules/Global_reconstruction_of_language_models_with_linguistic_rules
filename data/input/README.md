please find the input data here:

www. ... .de

download the input data, unzip the file 'input.zip' and place the four contained .json files into this folder '.../data/input/'.

These input datasets have to be processed as described in the following in order to use the data for this code project.

Since we do not have the rights to publish the review data, we provide the datasets withouth the review data (i.e., the tokens of the sentences are removed here). Therefore, the original datasets from yelp and amazon (cf. below) have to be used. For tokenization of the reviews to sentences and then the sentences to tokens, we used the python package NLTK. Then you can replace the placeholder '##put_token_<token_number>_here##' with the corresponding token with token number <token_number> in our input data. Here, the matching between our input data and the original review datasets is possible via the ids of the reviews and sentences (sentence id is just the enumeration of sentences in a review, in the following referred to as 'sentence_number'):
- yelp: the id of review sentences in our data is as follows: 'review_id||##||sentence_number' (review_id is the review id in the yelp dataset challenge)
- amazon: the id of review sentences in our data is as follows: 'reviewerID||##||asin||##||sentence_number' (reviewerID is the user id of the reviewer and asin is the item id of the reviewed item in the amazon dataset)

Finally, it is necessary to change the name of the four .json files. Here, the substring '_without_sentences' has to be removed. For example, 'laptops_amazon_ae_oe_150k_without_sentences.json' has to be renamed to 'laptops_amazon_ae_oe_150k.json'

---------

The original datasets from yelp and amazon containing the reviews are:
- the yelp dataset challenge (cf. www.yelp.com/dataset; downloaded January 2021)
- Amazon as provided by Ni, J., Li, J., & McAuley, J. (2019, November). Justifying recommendations using distantly-labeled reviews and fine-grained aspects. In Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP) (pp. 188-197). cf. https://cseweb.ucsd.edu/~jmcauley/datasets.html
