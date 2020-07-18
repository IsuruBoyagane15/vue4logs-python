from configs import *
from library.all_experiments import *

import numpy as np
import scipy.special
import pandas as pd
from os.path import join as join_path

dataset = ALL_EXPERIMENTS[experiment_nr]
experiment_sub_dir = experiment_type + "_" + str(epochs)
results_ = "results/" + experiment_sub_dir
experiment_dir = str(experiment_nr) + "_" + dataset


def getTemplate(candidate):
    # candidate: list of list
    abstraction = ''

    # transpose row to column
    candidate_transpose = list(zip(*candidate))
    candidate_length = len(candidate)

    if candidate_length > 1:
        # get abstraction
        abstraction_list = []
        for index, message in enumerate(candidate_transpose):
            message_length = len(set(message))
            if message_length == 1:
                abstraction_list.append(message[0])
            else:
                abstraction_list.append('<*>')

        abstraction = ' '.join(abstraction_list)

    elif candidate_length == 1:
        abstraction = ' '.join(candidate[0])

    return abstraction


"""**Read Ground ruth templates**"""

df_raw_logs = pd.read_csv("gt/" + dataset + "_2k.log_structured.csv")
raw_logs = df_raw_logs['Content'].to_numpy()


"""**Get best clusterong based config**"""

len(raw_logs)

predicted_labels = np.loadtxt('results/' + experiment_sub_dir + '/18_apache/predicted_labels.csv', delimiter=',')

groups = pd.DataFrame()
groups['predicted_labels'] = predicted_labels
groups['raw_logs'] = raw_logs

for i in np.unique(groups.predicted_labels):
    print(i)
    for index, row in groups.iterrows():
        if row['predicted_labels'] == i:
            print(row['raw_logs'])
    print('\n')

clusters = {}
for i in range(0, len(predicted_labels)):
    try:
        clusters[predicted_labels[i]].append(raw_logs[i].split(" "))
    except KeyError:
        clusters[predicted_labels[i]] = [raw_logs[i].split(" ")]

output = {}
# text_report = ''
for cluster in clusters.items():
    template = getTemplate(cluster[1])
    output['E' + str(cluster[0])] = template
    print(template)
    # text_report += template
    for log in cluster[1]:
        print(log)
        # text_report += str(log)
    print('\n')
    # text_report += '\n'


def write_output(id_to_template):
    line = {
        'EventId': list(id_to_template.keys()),
        'EventTemplate': list(id_to_template.values())
    }

    df = pd.DataFrame(line)
    df.to_csv(join_path(results_, experiment_dir, dataset + '_templates.csv'))


write_output(output)

structured_log = df_raw_logs.drop(['EventTemplate', 'EventId'], axis=1)
df_eventTemplates = pd.read_csv(join_path(results_, experiment_dir, dataset + '_templates.csv'))

template = []
eventId = []

for label in predicted_labels:
    template.append(df_eventTemplates[df_eventTemplates['EventId'] == 'E' + str(label)]['EventTemplate'])
    eventId.append(df_eventTemplates[df_eventTemplates['EventId'] == 'E' + str(label)]['EventId'])

structured_log['EventId'] = eventId
structured_log['EventTemplate'] = template

structured_log.to_csv(join_path(results_, experiment_dir, dataset + '_structured.csv'))


def evaluate(ground_truth, parsed_result):
    df_ground_truth = pd.read_csv(ground_truth)
    df_parsed_log = pd.read_csv(parsed_result)

    # Remove invalid ground truth event Templates
    null_log_ids = df_ground_truth[~df_ground_truth['EventTemplate'].isnull()].index
    df_ground_truth = df_ground_truth.loc[null_log_ids]
    df_parsed_log = df_parsed_log.loc[null_log_ids]

    (precision, recall, f_measure, accuracy) = get_accuracy(df_ground_truth['EventTemplate'],
                                                            df_parsed_log['EventTemplate'])
    print('Precision: %.4f, Recall: %.4f, F1_measure: %.4f, Parsing_Accuracy: %.4f' % (
        precision, recall, f_measure, accuracy))
    return f_measure, accuracy


def get_accuracy(series_ground_truth, series_parsedlog, debug=False):
    series_ground_truth_value_counts = series_ground_truth.value_counts()
    real_pairs = 0
    for count in series_ground_truth_value_counts:
        if count > 1:
            real_pairs += scipy.special.comb(count, 2)

    series_parsed_log_value_counts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsed_log_value_counts:
        if count > 1:
            parsed_pairs += scipy.special.comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0  # determine how many lines are correctly parsed
    for parsed_eventId in series_parsed_log_value_counts.index:
        log_ids = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_ground_truth_logId_value_counts = series_ground_truth[log_ids].value_counts()
        error_eventIds = (parsed_eventId, series_ground_truth_logId_value_counts.index.tolist())
        error = True
        if series_ground_truth_logId_value_counts.size == 1:
            ground_truth_event_id = series_ground_truth_logId_value_counts.index[0]
            if log_ids.size == series_ground_truth[series_ground_truth == ground_truth_event_id].size:
                accurate_events += log_ids.size
                error = False
        if error and debug:
            print('(parsed_eventId, ground_truth_event_id) =', error_eventIds, 'failed', log_ids.size, 'messages')
        for count in series_ground_truth_logId_value_counts:
            if count > 1:
                accurate_pairs += scipy.special.comb(count, 2)

    precision = float(accurate_pairs) / parsed_pairs
    recall = float(accurate_pairs) / real_pairs
    f_measure = 2 * precision * recall / (precision + recall)
    accuracy = float(accurate_events) / series_ground_truth.size
    return precision, recall, f_measure, accuracy


print(evaluate("gt/" + dataset + "_2k.log_structured.csv",
               results_ + "/" + experiment_dir + "/" + dataset + '_structured.csv'))
