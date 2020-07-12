# Tensorflow deep learning punctuation corrector

Simple project, used for correcting english text punctuation correcting, made for learing purposes.

### Project contains next modules:
 - corrector_dataset_builder - preprocessing sentences, building dataset for training
 - encdec_model - for now, contains only predictor.py, that is used for making prediction
 - encdec_model_builder - contains models builders
 - flask_app - simple flask app that is used for fast checking of model predictions
 - main - used for dataset building, training, and fast checking
 - tests - unit-tests made with pytest
 - utils - simple utilization functions
 
### Dataset
Dataset is grammarly and punctuatially correct english sentences from https://tatoeba.org/

### Training data structure
Model is training with 3 dataset: 
- input1 - sentence with puctuation mistakes
- input2 - correct sentence with start token, used for decoder teacher forcing
- target - correct sentence with end token, used as target data

All sentences is lowercase, but in input2 and target datasets there is uppercase token before any must-be-uppercase characters

### Model
Default model is seq2seq encoder-decoder models, with lstm and simple attention. Teacher forcing is used for decoder training.

### Predictions
Predictions are made with EncDecPredictor class. Predictor will use encoder for getting state [hidden and cell state] and sequence, then will itterate until end_token predicted, each itteration decoder state will be saved and re-used. This is usuall encoder-decoder realization.