import numpy as np
import pandas as pd
import mne
import os




all_data = []
all_sentence_ids = []

folder = "C:\\Users\\cally\\ProgramsAndProjects\\NN Hwk\\Project\\eegs"

for f in os.listdir(folder):
    file = os.path.join(folder, f)

    participant = f.split('_')[0]
    df = pd.read_csv(f"C:\\Users\\cally\\ProgramsAndProjects\\NN Hwk\\Project\\events\\{participant}_events.csv")


    epochs = mne.read_epochs(file, preload=True)

    id_to_label = {v: k for k, v in epochs.event_id.items()}

    epoch_idx = 0
    csv_idx = 0

    while epoch_idx < len(epochs) and csv_idx < len(df):
    # Get current epoch label
        epoch_label = id_to_label[epochs.events[epoch_idx, 2]]

    # Get current CSV row event_name
        csv_label = df.iloc[csv_idx]['event_name']

        if epoch_label == csv_label:
        # match: append data and sentence_id
            all_data.append(epochs[epoch_idx].get_data())
            all_sentence_ids.append((df.iloc[csv_idx]['sentence_id'],df.iloc[csv_idx]['target']))

            epoch_idx += 1
            csv_idx += 1
        else:
        # no match: advance only the CSV pointer
            csv_idx += 1



if all_data:
    all_data = np.concatenate(all_data, axis=0)

print(all_data.shape)
print(len(all_sentence_ids))


np.savez(r"C:\\Users\\cally\\ProgramsAndProjects\\NN Hwk\\Project",
         data=all_data,
         sentence_ids=all_sentence_ids)


