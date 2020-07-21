from configs import *
from library.all_experiments import *
from sklearn.cluster import Birch
from sklearn.metrics import homogeneity_completeness_v_measure, silhouette_score

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

dataset = ALL_EXPERIMENTS[experiment_nr]

experiment_sub_dir = experiment_type + "_" + str(epochs)
experiment_dir = str(experiment_nr) + "_" + dataset
result_dir = "results/" + experiment_sub_dir

embedding_file = os.path.join("results", experiment_sub_dir, experiment_dir,
                              dataset + "_" + experiment_sub_dir + "_embeddings.csv")
ground_truth_file = "gt/" + dataset + "_2k.log_structured.csv"
cluster_result_file = os.path.join("results", experiment_sub_dir, experiment_dir,
                                   "clustering_results.txt")
predicted_labels_file = os.path.join("results", experiment_sub_dir, experiment_dir,
                                     "predicted_labels.csv")

if os.path.exists(result_dir + "/" + experiment_dir + "/" + dataset + "_" + experiment_sub_dir + "_embeddings.csv"):
    X = pd.read_csv(result_dir + "/" + experiment_dir + "/" + dataset + "_" + experiment_sub_dir + "_embeddings.csv",
                    header=None).to_numpy()
else:
    print("No embedding file")
    exit(0)

df = pd.read_csv(ground_truth_file)
eventIds = df["EventId"]
event_id_list = []
for eventId in eventIds:
    event_id_list.append(int(eventId[1:]))
clusters = np.array(event_id_list)

step = 0.2
v_measures = []
silhouette_measures = []
thresholds = []
current_threshold = 0

while True:
    brc = Birch(threshold=current_threshold, branching_factor=50, n_clusters=None, compute_labels=True, copy=True)
    brc.fit(X)
    labels = brc.predict(X)

    v_measure = homogeneity_completeness_v_measure(clusters, labels)[2]
    try:
        silhouette_measure = silhouette_score(X, labels)
    except ValueError:
        print("Data is clustered into one cluster")
        break

    thresholds.append(current_threshold)
    v_measures.append(v_measure)
    silhouette_measures.append(silhouette_measure)
    current_threshold += step

plt.plot(thresholds, v_measures, label="v-measure")
plt.plot(thresholds, silhouette_measures, label="silhouette")

plt.xlabel('threshold')
plt.legend()
plt.show()

selected_threshold = thresholds[silhouette_measures.index(max(silhouette_measures))]
print("Threshold giving highest v-measure :", selected_threshold)
brc = Birch(threshold=selected_threshold, branching_factor=50, n_clusters=None, compute_labels=True, copy=True)
brc.fit(X)
predicted_labels = brc.predict(X)

max_silhouette_score = max(silhouette_measures)
max_v_measure = max(v_measures)

print("Maximum v-measure :", max_v_measure)
print("Maximum silhouette score :", max_silhouette_score)
print("Selected threshold :", selected_threshold)
print("Predicted labels :", predicted_labels)

cluster_count = len(np.unique(predicted_labels))
print("Cluster count :", cluster_count)

with open(cluster_result_file, "w") as c_file:
    c_file.write("max_v_measure :" + str(max_v_measure) + "\n")
    c_file.write("max_silhouette_score :" + str(max_silhouette_score) + "\n")
    c_file.write("selected_threshold :" + str(selected_threshold) + "\n")
    c_file.write("cluster_count :" + str(cluster_count) + "\n")

np.savetxt(predicted_labels_file, predicted_labels, delimiter=",")
