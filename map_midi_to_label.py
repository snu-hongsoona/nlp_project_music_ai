import csv
from collections import OrderedDict, defaultdict
import numpy as np
import json
from scipy.stats import gaussian_kde
from tqdm import tqdm

LABEL_LIST = ["Stable beat", "Mechanical Tempo", "Intensional", "Regular beat change", "Long", "Cushioned", "Saturated (wet)", "Clean", "Subtle change", "Even", "Rich", "Bright", 
"Pure", "Soft", "Sophisticated(mellow)", "balanced", "Large range of dynamic", "Fast paced", "Flowing", "Swing(Flexible)", "Flat", "Harmonious", "Optimistic(pleasant)", "HIgh Energy", 
"Dominant(forceful)", "Imaginative", "Ethereal", "Convincing"]
LABEL_MAP = {i: label for i, label in enumerate(LABEL_LIST)}
PIANIST_MAP = OrderedDict()

file = open('total.csv', encoding="utf-8")
#file = open('total.csv', encoding="utf-8")

def estimate_maxima(data):
    if len(set(data))<=1: # all datas are equal
        return data[0]
    kde = gaussian_kde(data)
    no_samples = 50
    samples = np.linspace(min(data), max(data), no_samples)
    probs = kde.evaluate(samples)
    #maxima_index = probs.argmax()
    # in case if more than 1 argmaxs
    winner = np.argwhere(probs == np.amax(probs))
    maxima = np.average(samples[winner.flatten()])
    return maxima

def midi_label_map_apex():

    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)

    rows = []
    for row in csvreader:
        rows.append(row)

    # sort by each segments
    music_label_map = defaultdict(list)
    for row in rows:
        user = row[0]
        file_name = row[2].split(".")[0]
        #label_row = row[3:-2]
        label_row = row[6:-2] # skip 1-1 ~ 1-3
        for idx, elem in enumerate(label_row):
            if elem == "":
                label_row[idx] = 0.0
            else:
                label_row[idx] = float(elem)
        # skip 0
        if 0.0 in label_row:
            continue
        else:
            music_label_map[file_name].append(label_row)

    music_label_map_apex = dict()

    # kernel density estimation
    for key, annot_list in tqdm(music_label_map.items()):
        annot_list = np.array(annot_list).transpose()
        maxima = np.array([estimate_maxima(row)/7 for row in annot_list])
        maxima = maxima.transpose().tolist()
        music_label_map_apex[key] = maxima

    # add pianist info
    for key, annot_list in tqdm(music_label_map_apex.items()):
        if key.split("_")[-2] not in PIANIST_MAP:
            PIANIST_MAP[key.split("_")[-2]] = len(PIANIST_MAP)
    print(PIANIST_MAP)
    
    for key, annot_list in tqdm(music_label_map_apex.items()):
        music_label_map_apex[key].append(PIANIST_MAP[key.split("_")[-2]])

    json.dump(music_label_map_apex, open("midi_label_map_apex_reg_cls.json", 'w'))

# def midi_label_map_apex_filtered() : # 0922

#     filtered_label_list = ["Rich", "balanced", "Large range of dynamic", "Swing(Flexible)", "Flat", "Harmonious", "HIgh Energy",
#                            "Dominant(forceful)", "Imaginative", "Ethereal", "Convincing"]
#     filtered_loc = [LABEL_LIST.index(name) for name in filtered_label_list]

#     csvreader = csv.reader(file)
#     header = []
#     header = next(csvreader)

#     rows = []
#     for row in csvreader:
#         rows.append(row)

#     # sort by each segments
#     music_label_map = defaultdict(list)
#     for row in rows:
#         user = row[0]
#         file_name = row[2].split(".")[0]
#         label_row = row[3:-2]
#         label_row = [label_row[loc] for loc in filtered_loc]
#         for idx, elem in enumerate(label_row):
#             if elem == "":
#                 label_row[idx] = 0.0
#             else:
#                 label_row[idx] = float(elem)
#         # skip 0
#         if 0.0 in label_row:
#             continue
#         else:
#             music_label_map[file_name].append(label_row)

#     music_label_map_apex = dict()

#     # kernel density estimation
#     for key, annot_list in tqdm(music_label_map.items()):
#         annot_list = np.array(annot_list).transpose()
#         maxima = np.array([estimate_maxima(row)/7 for row in annot_list])
#         maxima = maxima.transpose().tolist()
#         music_label_map_apex[key] = maxima

#     json.dump(music_label_map_apex, open("data/xai/midi_label_map_apex_filtered.json", 'w'))


# def midi_label_map_apex_except_filtered() : # 0922 # 13 labels

#     filtered_label_list = ["Rich", "balanced", "Large range of dynamic", "Swing(Flexible)", "Flat", "Harmonious", "HIgh Energy",
#                            "Dominant(forceful)", "Imaginative", "Ethereal", "Convincing"]
#     filtered_loc = [LABEL_LIST.index(name) for name in filtered_label_list]
#     filtered_loc = [loc for loc in list(range(0,28)) if loc not in filtered_loc]

#     csvreader = csv.reader(file)
#     header = []
#     header = next(csvreader)

#     rows = []
#     for row in csvreader:
#         rows.append(row)

#     # sort by each segments
#     music_label_map = defaultdict(list)
#     for row in rows:
#         user = row[0]
#         file_name = row[2].split(".")[0]
#         label_row = row[3:-2]
#         label_row = [label_row[loc] for loc in filtered_loc]
#         label_row = label_row[4:]
#         for idx, elem in enumerate(label_row):
#             if elem == "":
#                 label_row[idx] = 0.0
#             else:
#                 label_row[idx] = float(elem)
#         # skip 0
#         if 0.0 in label_row:
#             continue
#         else:
#             music_label_map[file_name].append(label_row)

#     music_label_map_apex = dict()

#     # kernel density estimation
#     for key, annot_list in tqdm(music_label_map.items()):
#         annot_list = np.array(annot_list).transpose()
#         maxima = np.array([estimate_maxima(row)/7 for row in annot_list])
#         maxima = maxima.transpose().tolist()
#         music_label_map_apex[key] = maxima

#     json.dump(music_label_map_apex, open("data/xai/midi_label_map_apex_except_filtered.json", 'w'))

if __name__ == "__main__":
    #midi_label_map_apex_filtered() 
    #midi_label_map_apex_except_filtered() 
    midi_label_map_apex() 