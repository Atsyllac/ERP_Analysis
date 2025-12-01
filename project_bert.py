from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler


npz = np.load("C:\\Users\\cally\\ProgramsAndProjects\\NN Hwk\\Project\\englishsentences&stringlabels.npz")
english_sentences = npz["sentences_english"].reshape(-1).tolist()


tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")

# #testing to make sure a semantically and grammatically incorrect sentence will still be tokenized
# test = 'Авторы получали районах'
# print(tokenizer.tokenize(test))

inputs = tokenizer(
    english_sentences,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
)

with torch.no_grad():
    outputs = model(**inputs)

hidden_states = outputs.last_hidden_state #shape batch, token seq len, hidden size

oneD_token_features = hidden_states.transpose(1, 2).detach().numpy()  # shape (B, H, T) for conv1D
# twoD_token_features = oneD_token_features.unsqueeze(1) #shape (B, 1, H, T) for conv2D

print(oneD_token_features.shape)
print(oneD_token_features[0])

str_labels = npz["labels_str"]
encoder = LabelEncoder()
labels = encoder.fit_transform(str_labels)
print(labels[0:15])

'''
grammar: 0
normal: 1
semantic: 2
sem_gram: 3
'''

npz2 = np.load("C:\\Users\\cally\\ProgramsAndProjects\\NN Hwk\\Project\\eegdata&allsentences.npz", mmap_mode='r')
eeg_data = npz2["data"]


split = int(0.8 * len(eeg_data))
eeg_train, eeg_test = eeg_data[:split], eeg_data[split:]
sentence_train, sentence_test = oneD_token_features[:split], oneD_token_features[split:]
labels_train, labels_test = labels[:split], labels[split:]

print(eeg_train.shape, sentence_train.shape, labels_train.shape)
print(type(eeg_train), type(sentence_train), type(labels_train))


np.savez(r"C:\\Users\\cally\\ProgramsAndProjects\\NN Hwk\\Project",
         eeg_train=eeg_train,
         eeg_test=eeg_test,
         sentence_train=sentence_train,
         sentence_test=sentence_test,
         labels_train=labels_train,
         labels_test=labels_test
         )

