    Simple english language corrector, made with tensorflow keras, lstm/gru, attention encoder-decoder model. 

    Project containing next modules:

     1. corrector_dataset_builder - processing and building datasets from numerous pucntuationally correct sentences
        Each sentence needs to be processed through the next pipeline:
        Example sentence is "I'm sorry, I can't stay long."

        1. All elements must be padded: 
        "I'm sorry , I can't stay long ."

        2. Mistakes must be generated: punctuations symbols must be removed, uppercase must be lowercase
        "i'm sorry i can't stay long"

        3. For model fitting, there shoul be 3 datasets:
        input1 - processed sentence with mistakes - "i'm sorry i can't stay long"
        input2 - original, correct padded sentence with start token before sentence, used for teacher forcing - "<s> I'm sorry , I can't stay long ."
        target - original, correct padded sentence with end token after sentence - "I'm sorry , I can't stay long . <e>"

        4. All datasets (input1, input2, target) must be tokenized and padded with tf.keras pad_sequence, tokenizer must be fitted on target dataset but with start token before target sentence  "<s> I'm sorry , I can't stay long . <e>"

    2. encdec_model_builder - building different models, returning composite, encoder and decoder models:
        Composite model is used for training 
        Encoder-decoder pair used for predicting 
    
    3. encdec_model - different functions for model, like loading/saving, predicting etc.

    4. flask_app - simple app for correcting sentences in browser

    5. tests - unit tests made with pytest

    main.py is used for building datasets, training and saving

