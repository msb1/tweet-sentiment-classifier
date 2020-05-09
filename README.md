# tweet-sentiment-classifier
 Tweet Sentiment Classifier trained in 1.6M Kaggle Dataset
 
<ol>
<li>Classifier text preview analysis:
    <ul>
    <li>Text cleaning: (i) remove punctuation except for ! and * and ' as these appear to convey meaning in "Tweet Slang'; (ii) remove all numbers; (iii) remove all
    stopwords (manually from Gensim stop words list); (iv) remove all entities (hashtags and users) and url's. Note using Spacy to remove stop words and all punctuation
    resulted in loss of meaning so the this was done manually. Some stop words appear to convey meaning in 'Tweet Slang'</li>
<li>Determine distribution (histogram) of tweet word lengths to set max length for tweet training features in neural net models</li>
<li>Create sorted word dictionary for words contained in 1.6M tweets; sorting is done by frequency of occurence. Also separate into dictionaries for
words contained in positive and negative tweets to assess correlation (if any)</li>
<li>Plot Word Clouds for positive and negative words from dictionaries. This displays most used words in tweets.</li>
<li>Plot bar char of positive and negative words from dictionaries</li>
<li>Plot word correlation between words in positive and negative dictionaries</li>
<li>Create Tokenization and Label Index Dicts and save to files using Keras Tokenizer. Save Tokenizer dictionary to file.</li>
<li>Tokenize each row in training dataset</li>
<li>Truncate/Pad each row to fixed length and Create Label (with 0 or 1 value) per Row for Training BCE for each label</li>
</ul>
</li>
<li>Train bidirectional LSTM model with dropout and L2 regularization (to avoid overfitting). Note that the model is set to create word embeddings
rather than use pre-trained embeddings (from Spacy or Glove) as is typically done in Sentiment Analysis. This is because 'Tweet Slang' often has different
meaning and different neighboring words and entities than standard English text.</li>
<li>An 80/20 train/validation split is performed on shuffled tweets.</li>
<li>To further validate the training, a 10-fold cross validation is performed</li>
</ol>
