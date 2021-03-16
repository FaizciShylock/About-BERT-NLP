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
    
   
