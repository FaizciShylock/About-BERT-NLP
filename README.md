# About-BERT-NLP

BERT: Bidirectional Encoder Representations from Transformers

It is a technique which was developed for NLP by the researchers at Google AI Language
It received a lot of attention in Deep Learning community due to its incredible performance.

# Why was BERT needed ?
  
  # Lack of enough data
    - In order to perform well, deep learning based NLP models require much larger amounts of data
    - Researchers have developed various techniques for training general purpose language representation models using enormous piles of unannotated text on the web (pre-training)
    - These general purpose pre-trained models can be then fine-tuned on smaller task-specific datasets
    - BERT is a recent addition to these techniques for NLP pre-training and it presented very successful results in a wide variety of NLP tasks
  
  # Usage
    - BERT models can be used to extract high quaity language features from our text data
    - BERT models can be fine-tuned on a specific task (e.g. sentiment analyss, question answering)

# What is the core idea behind it ? 

  # Language models and "fill in the blank" task

    Before BERT, a language model would have looked at a text sequence during training from either left-to-right or combined left-to-right and right-to-left
    This "one-directional" approach works well for generating sentences. (word by word completion)
    
    However, BERT is a bidirectionally trained language model and its key techincal innovation comes from "bi-directional training" concept.
    Instead of predicting the next word in a sequence, BERT uses a technique called Masked LM (MLM).
    
  # Masked LM
    
     It randomly masks words in the sentence and then it tries to predict them.
     Masking: the model looks in both directions and it uses the full context of the sentence both left and right surroundings
     
  # Difference from other models
    
    It takes both the previous and next tokens into acount at the same time.
    Existing LSTM based models were missing this "same-time" part
    It might be considered as "non-directional approach"
   
  # Why this non-directional approach is so successful ?
  
    Pre-trained langauge representations can either be context-free or context-based.
    Context-based representations can be "unidirectional" or "bidirectional".
    Context-free models like word2vec generate a single word embedding rep. for each word in the vocabulary
  
  # Transformer Model Architecture
    
    BERT is based on the transformer model architecture, instead of LSTMs
    A Transformer works by performing a small, constant number of steps
    In each step, it applies an attention mechanism to understand relationships between all words in a sentence, regardless of their respective position
 
# How does it work ?
    BERT relies on a Transformer which is an attention mechanism that learns contextual relationships between words in a text
    A basic Transformer consists of an encoder and a decoder.
    Since BERT's goal is to generate a language representation model, it only needs the encoder part.
    
    The input to the encoder for BERT is a sequence of tokens, which are first converted into vectors and then processed in the neural network
    Before processing can start, the input should be massaged and decorated with some extra metadata:
      - Token embeddings
      - Segment embeddings
      - Positional embeddings
    
    The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings
    
  # Masked LM (MLM)
    Randomly mask out 15% of the words in the input, replacing them with a MASK token, and then run the entire seuqence through the BERT attention based encoder and then predict only the masked words, based on the context provided by the other non-masked words int he sequence

    Out of the 15% of the tokens selected for masking:
      80% of the tokens are actually replaced with the token MASK
      10% of the time tokens are replaced with a random token
      10% of the time tokens are left unchanged
      
    While training the BERT loss function considers only the prediction of the masked tokens  and ignores the others.
    This results in a model that converges much more slowly than LSTMs
  
  # Next Sentence Prediction (NSP)
    In order to understand relationship between two sentences, BERT training process also uses next sentence prediction
    It is relevant for tasks like question answering.
    During training the model gets as input pairs of sentences and it learns to predict if the second sentence is the next sentence in the original text as well
    
    BERT separates sentences with SEP token
    
    During training the model is fed with two input sentences at a time:
      - 50% of the time the 2nd sent. comes after the first
      - 50% of the time it is a random sent. from the full corpus
    
    Then, BERT is required to predict whether the second sentence is random or not, with the assumption that the random sentence will be disconnected from the first sentence
  
!!! The model is trained with both Masked LM and Next Sentence Prediction together.:
This is to minimize combined loss function of the two strategies - "together is better"

  # Architecture
    There are four types of pre-trained version of BERT depending on the scale of the model architecture:
      - BERT-Basse
      - BERT-Large
   
