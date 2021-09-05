NEGATE = \
    ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]
VADER_LEXICON_PATH = "/home/madhu/vaderSentiment/vaderSentiment/vader_lexicon.txt"

def negated(input_words, include_nt=True):
    """
    Returns the number of negation words in the input
    """
    input_words = [str(w).lower() for w in input_words]
    neg_words = []
    neg_words.extend(NEGATE)
    count = 0
    for word in neg_words:
        if word in input_words:
            count += 1
    if include_nt:
        for word in input_words:
            if "n't" in word:
                count += 1
    return count
    
def read_vader_sentiment_dict(filepath=None):
    if filepath == None:
        filepath = VADER_LEXICON_PATH
    vader_sentiment_scores = {}
    with open(filepath, "r") as fin:
        for line in fin:
            values = line.split("\t")
            vader_sentiment_scores[values[0]] = float(values[1])

    return vader_sentiment_scores