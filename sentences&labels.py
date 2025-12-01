import numpy as np
import pandas as pd


npz = np.load("C:\\Users\\cally\\ProgramsAndProjects\\NN Hwk\\Project\\Project.npz")
eeg_sentence_ids = npz['sentence_ids']

stimuli = pd.read_csv("C:\\Users\\cally\\ProgramsAndProjects\\NN Hwk\\Project\\stimuli.csv")
stimuli['id'] = stimuli['id'].astype(int)

sentences = []
labels = []


#pulls the correct sentences from the csv using the id and label, puts the label in one list and sentence in another
for id, target in eeg_sentence_ids:
    id = int(id)

    match = (stimuli['id'] == id) & (stimuli['target'] == target)

    matched = stimuli.loc[match, 'sentence'].tolist()
    sentences.append(matched)
    labels.append(target)

sentences = np.array(sentences)
labels = np.array(labels)

print(sentences.shape) #shape (11162, 1)
print(labels.shape)
print(labels[0:5])

np.savez(r"C:\\Users\\cally\\ProgramsAndProjects\\NN Hwk\\Project",
         sentences_english=sentences,
         labels_str=labels)



