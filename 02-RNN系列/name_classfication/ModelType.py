from enum import Enum

class ModelType(str,Enum):
    RNN = "rnn"
    LSTM = "lstm"
    GRU = "gru"