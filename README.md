# Linguistic ERP Analysis
Repo for final project of CSCI 5122 (Neural Networks &amp; Deep Learning)

## Abstract
In neuroscience, Event-related potentials (ERPs) are very small voltages generated in the brain structures in response to specific events or stimuli (Blackwood and Muir, 1990). These are measured using electroencephalography (EEG). There are two widely-studied ERPs related to linguistics: the N400 and the P600. The N400 is a negative waveform that peaks about 400ms after a person is introduced to a lexical or semantic error. The P600 waveform peaks positively about 600ms after a stimulus containing a syntactical error.
In this project, I use two CNNs and an ANN to see if, given sentence-level embeddings and sentence-locked EEG data, an ANN can classify whether a sentence contains a grammatical, syntactical, or syntactic_grammatical error, or is simply normal. While reasearch has been done on using CNNs to decode EEGs (see EEGNet and DeepConvNet), I have not seen a classification task like this one, especially related to non-English data.


## Analysis
Because I am using both sentence embeddings and spatial data, I branched my model architecture. I use the (Semantic and Inferred Grammar Neurological Analysis of Language) SIGNAL dataset (https://huggingface.co/datasets/ContributorsSIGNAL/SIGNAL) for my project.

### Data Preparation
The SIGNAL corpus contains EEG data from 21 participants, 600 sentences per participant. Using the MNE library in python I removed noisy data and gathered a total of 11,162 samples. Each EEG epoch/trial contained 63 channels (number of EEG electrodes on the person's scalp, each providing its own signal) and 1201 timepoints (taken sequentially as the participant read each sentence). Thus, the EEG data has shape (11162, 62, 1201).

Sentences and their respective target stimuli were provided with the data. I separated the sentences from the stimuli labels and used contextual embeddings via RuBERT (https://huggingface.co/DeepPavlov/rubert-base-cased)--sentences are in Russian. I did not use static embeddings nor did I pool the embedding data, as I wanted the encoder to capture the incongruency in the choices of words used for the incorrect sentences. The shape of this data is (11162, 768, 10) ((n, h, t)). 

As the .npz files containing the total eeg data, total embedding data, and total label data are too large for me to upload to github, I will simply upload the code I used to prepare the data in the repo.

### Models
Rather than creating two sequential models, I used the Keras's functional API seen here https://pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/

#### The EEG CNN Branch
Due to the 2D spatio-temporal nature of the EEG data, I modeled it using a CNN with 2D convolution. Kernel sizes were different from the norm, as a taller kernel would capture EEG signals more faithfully.

#### The Sentence Embedding CNN Branch
I used a CNN for the embeddings as well, though this time with only 1D convolution. Kernel sizes and pooling sizes were standard.

#### The ANN Classifier
Lastly, I concatenate the outputs from the two CNN models. I started with gated fusion so the outputs from each model had their own weights and the ANN did not prioritize both equally, but found that this did not increase the model's accuracy and slowed training. The ANN itself is quite simple, with the output being the sentence/EEG grammaticality classification.


## Results



