# tweet-sentiment-classifier
<h6>Two Tweet Sentiment Classifiers are trained on 1.6M Kaggle Dataset including a BiLSTM in TensorFlow/Keras and a DistilBert Transformer in PyTorch</h6>
<p>Note that three transformers (Bert, DistilBert, and RoBerta) where tested in Tensorflor/Keras and all of them stopped with an internal BLAS error. 
Therefore PyTorch was tested, did not show any internal errors and was used for this model
development. As these transformers from HuggingFace are relatively new to Tensorflow/Keras,
 hopefully these errors will be corrected in the near future.</p>
 
<h6>Model Development Summary</h6>
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
<li>Create Tokenization and Label Index Dicts and save to files using Keras and DistilBert Tokenizers for the BiLSTM and Transformer models, respectively. 
Save Keras Generated Tokenizer dictionary to file.</li>
<li>Tokenize each row in training dataset</li>
<li>Truncate/Pad each row to fixed length and Create Label (with 0 or 1 value) per Row for Training using binary cross entropy as loss function</li>
<li>For the DistilBert transformed model, generate an attention mask for each tweet/row</li>
</ul>
</li>
<li>An 80/20 train/validation split is performed on shuffled preprocessed tweets.</li>
<li>Train bidirectional LSTM model with dropout and L2 regularization (to avoid overfitting). Note that the model is set to create word embeddings
rather than use pre-trained embeddings (from Spacy or Glove) as is typically done in Sentiment Analysis. This is because 'Tweet Slang' often has different
meaning and different neighboring words and entities than standard English text. To further validate the training, a 10-fold cross validation is performed</li>
<li>Train DistilBert transformer model with two dense/linear output layers and dropout. DistilBert embeddings are used for the model.</li>
</ol>

<h6>Note that the DistilBert Transformed appears to outperform the BiLSTM model as expected. However it would be nice to perform a cross validation
on Transformer model. This will be skipped at this time since the transformer model training took approximately 58hrs on GTX1060 GPU. Additional 
Transformer modeling is planned to RoBerta since it is purported to be best in class. Based on existing benchmarks, it is expected
that training a Roberta Transformer for the Tweet Sentiments will take approx. 100hrs on the same GPU.</h6>
