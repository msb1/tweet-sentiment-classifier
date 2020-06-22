# tweet-sentiment-classifier
<h6>Three Tweet Sentiment Classifiers are trained on 1.6M Kaggle Dataset including a BiLSTM in TensorFlow/Keras, a BiLSTM with Multihead (self) Attention and a DistilBert Transformer in PyTorch</h6>
<p>Note that three transformers (Bert, DistilBert, and RoBerta) were tested in Tensorflor/Keras and all of them stopped with an internal BLAS error. 
Therefore the model was rewritten and run with PyTorch where it did not exhibit any internal errors or problems. These transformers from HuggingFace are relatively new to Tensorflow/Keras and hopefully these errors will be corrected in the near future.</p>
 
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
<li>The two biLSTM models both use dropout and L2 regularization (to reduce overfitting tendencies). These models are also setup </li>
<li>Models include:
<ul>
<li>Tensorflow/Keras bilstm model with return sequences set to false so that only the last output state is returned from the layer; this is then input to two fully connected layers preceded by dropout layers. </li>
<li>Tensorflow/Keras bilstm model with return sequences set to true so that all hidden states and the output state are returned from the layer; this is then input into a multihead attention followed by two global pooling layers (concatenated) and fully connected layers, the latter preceded by a dropout layer. </li>
<li>DistilBert transformer model with two dense/linear output layers and dropout. DistilBert embeddings are used for the model.</li>
</ul>

<h6>Note that the DistilBert Transformer and the biLSTM with self attention appear to outperform the biLSTM model as expected by about 1 - 2% in terms of accuracy. 
The training of DistilBert transformer model (the smallest and simplest of all the Hugging Face models) training took approximately 58hrs on GTX1060 GPU. Both biLSTM and biLSTM with multihead attention took approximately 9 hours; the biLSTM with attention can also be run on CPU where the transformer model needs an expensive GPU. Therefore the biLSTM with attention should be used for this type of nlp classification unless it can be demonstrated that the transformer drastically outperforms it. Cost is far lower to run these models in production of CPU rather than GPU/TPU; scaling and load balancing is also far more readily accomplished with apps that can be run on CPU only.</h6>

<h6>
Note that transformer modeling was also coded in PyTorch and started for RoBertA (from Facebook) and XLNet (from Google). However, DistilBert (from Hugging Face) was chosen since it runs in about half the time on GPU and is shown to be 96% as effective when compared with BERT (from Google). The two transformer models with RoBertA and XLNet coded here were running approx 2X slower so they were not completed due to other deep learning priorities that needed GPU processing.
</h6>
