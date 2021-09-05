

NEGATE = \
    ["aint", "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt",
     "ain't", "aren't", "can't", "couldn't", "daren't", "didn't", "doesn't",
     "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither",
     "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
     "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
     "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
     "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
     "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite"]

def negated(input_words, include_nt=True, return_words_flag=False):
    """
    Return the number of negation words in the input
    """
    input_words = [str(w).lower() for w in input_words]
    neg_words = []
    neg_words.extend(NEGATE)
    count = 0    
    neg_words_in_input = []
    for word in neg_words:
        if word in input_words:            
            neg_words_in_input.append(word)
            count += 1
    if include_nt:
        for word in input_words:
            if "n't" in word:                
                neg_words_in_input.append(word)
                count += 1
    '''if "least" in input_words:
        i = input_words.index("least")
        if i > 0 and input_words[i - 1] != "at":
            return True'''

    if return_words_flag:
        return neg_words_in_input
    else:
        return len(neg_words_in_input)

def read_vader_sentiment_dict(filepath):
    vader_sentiment_scores = {}
    with open(filepath, "r") as fin:
        for line in fin:
            values = line.split("\t")
            vader_sentiment_scores[values[0]] = float(values[1])

    return vader_sentiment_scores