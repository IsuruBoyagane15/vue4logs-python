from library.all_imports import *
import numpy as np
from configs import *

experiment_sub_dir = experiment_type + "_" + str(epochs)
result_dir = "results/" + experiment_sub_dir

if not os.path.exists(result_dir):
    os.mkdir(result_dir)
    print(result_dir, "directory was created")

assert tf.__version__ == "1.0.1"
tf.set_random_seed(0)
np.random.seed(0)  # fix random seed for reproducibility
tf.logging.set_verbosity(tf.logging.INFO)

# Hyper parameter

from library.all_experiments import *

EXPERIMENT_ID = ALL_EXPERIMENTS[experiment_nr]
print("Running experiment: %s" % EXPERIMENT_ID)
EXPERIMENT_SPLIT_TOKEN = SPLIT_TOKEN[experiment_nr] if (experiment_nr in SPLIT_TOKEN.keys()) else SPLIT_TOKEN["default"]

unknown_token = g.unknown_token = "UNKNOWN_TOKEN"
log_line_start_token = g.logline_start_token = "LOG_START"
log_line_end_token = g.logline_end_token = "LOG_END"
pad_token = g.pad_token = "PAD_TOKEN"
g.vocabulary_max_lines = -1  # -1 means unlimited
g.max_line_len = 200

experiment_dir = "%.2d_%s" % (experiment_nr, EXPERIMENT_ID)
experiment_out_dir = join_path(result_dir, experiment_dir)

if not os.path.exists(experiment_out_dir):
    os.mkdir(experiment_out_dir)

# data files

datafile = g.datafile = "data/%s.log" % EXPERIMENT_ID
labels_true_file = "data/%s.ids" % EXPERIMENT_ID
processed_datafile = join_path(result_dir, experiment_dir, "%s_log.processed" % EXPERIMENT_ID)

test_data_processed = join_path(result_dir, experiment_dir, "%s_test_log.processed" % EXPERIMENT_ID)

VOCABULARY_FILE = g.VOCABULARY_FILE = join_path(result_dir, experiment_dir, "vocabulary.json")
hyper_parameter_file = join_path(experiment_out_dir, "hyperparams.json")

g.REGENERATE_VOCABULARY_FILES = True
g.REPROCESS_TRAININGS_DATA = True
learn_model = True

# Generate vocabulary

g.WORD_TO_INDEX_FILE = join_path(result_dir, experiment_dir, "word_to_index.json")
g.INDEX_TO_WORD_FILE = join_path(result_dir, experiment_dir, "index_to_word.json")
g.TOKENIZED_LOGLINES_FILE = join_path(result_dir, experiment_dir, "tokenized_log_lines.json")
g.SPLIT_TOKEN = EXPERIMENT_SPLIT_TOKEN

from library.vocabulary import *

print("Dataset fully loaded")

# Create the training data

if g.REPROCESS_TRAININGS_DATA:
    X_train = np.asarray([[word_to_index[w] for w in logline] for logline in tokenized_loglines])
    maximum_sequence_length = -1
    train_numbers_file = open(processed_datafile, "w")
    one_percent = len(X_train) / 100
    for i, log_line_as_word_id_sequence in enumerate(X_train):
        if i % one_percent == 0:
            print("Written line %i" % i)

        reversed_seq = list(reversed(log_line_as_word_id_sequence))
        sequence_length = str(len(log_line_as_word_id_sequence))
        if len(log_line_as_word_id_sequence) > maximum_sequence_length: maximum_sequence_length = len(
            log_line_as_word_id_sequence)  # find maximum sequence length
        word_id_seq = ",".join(map(str, log_line_as_word_id_sequence))  # encoder input: 1,2,3
        word_id_seq_reversed = ",".join(map(str, reversed_seq))  # decoder input: 3,2,1
        target_seq = ",".join(map(str, reversed_seq[1:] + [PAD_ID]))  # decoder target: 2,1,PAD

        signature_id = 5
        # write
        train_numbers_file.write(
            "%s|%s|%s|%s|%s\n" % (signature_id, sequence_length, word_id_seq, word_id_seq_reversed, target_seq))

    train_numbers_file.close()

    # tokenizing test data part
    tokenized_test_log_lines = []

    test_log_lines = list(open("data/test/%s.log" % EXPERIMENT_ID, 'r'))
    print("Loaded %i test_log_lines" % len(test_log_lines))
    total_lines = len(test_log_lines) if g.vocabulary_max_lines == -1 else g.vocabulary_max_lines
    print("Tokenizing test %i lines... " % total_lines)
    print("test_log_lines", len(test_log_lines))

    for i, test_log_line in enumerate(test_log_lines):

        test_log_line = test_log_line.lower()

        for char in g.SPLIT_TOKEN:
            test_log_line = test_log_line.replace(char, ' ' + char + ' ')

        tokenized_test_log_line = test_log_line.split(" ")[0:200]

        tokenized_test_log_line = [g.logline_start_token] + tokenized_test_log_line + [g.logline_end_token]

        tokenized_test_log_lines.append(tokenized_test_log_line)

        if g.vocabulary_max_lines > 0 and i > g.vocabulary_max_lines:
            break

    for i, test_log_line in enumerate(tokenized_test_log_lines):
        tokenized_test_log_lines[i] = [w if w in word_to_index else g.unknown_token for w in test_log_line]

    print("Tokenized logfile.")

    X_test = np.asarray([[word_to_index[w] for w in logline] for logline in tokenized_test_log_lines])
    maximum_sequence_length = -1
    test_numbers_file = open(test_data_processed, "w")
    one_percent = len(X_test) / 100

    for i, test_log_line_as_word_id_sequence in enumerate(X_test):
        if i % one_percent == 0: print("Written line %i" % i)

        reversed_seq = list(reversed(test_log_line_as_word_id_sequence))

        sequence_length = str(len(test_log_line_as_word_id_sequence))
        if len(test_log_line_as_word_id_sequence) > maximum_sequence_length: maximum_sequence_length = len(
            test_log_line_as_word_id_sequence)  # find maximum sequence length
        word_id_seq = ",".join(map(str, test_log_line_as_word_id_sequence))  # encoder input: 1,2,3
        word_id_seq_reversed = ",".join(map(str, reversed_seq))  # decoder input: 3,2,1
        target_seq = ",".join(map(str, reversed_seq[1:] + [PAD_ID]))  # decoder target: 2,1,PAD

        signature_id = 5

        # write
        test_numbers_file.write(
            "%s|%s|%s|%s|%s\n" % (signature_id, sequence_length, word_id_seq, word_id_seq_reversed, target_seq))
    test_numbers_file.close()
else:
    X_train = np.asarray([[word_to_index[w] for w in log_line] for log_line in tokenized_loglines])
    print("Training data has already been processed")

if not os.path.exists(labels_true_file):
    os.system("python3 create_true_labels.py -en %s" % EXPERIMENT_ID)

# Graph Hyper parameters

# TDB save vocabulary file
state_size = 256
batch_size = 200

num_examples_to_visualize = min(10000, len(tokenized_loglines))  # how many dots to show
num_examples_to_embed = len(
    tokenized_test_log_lines)

dropout_keep_probability = 0.7
num_lstm_layers = 1

DTYPE = tf.float32
num_samples = min(vocabulary_size, 500)  # number of samples to draw for sampled softmax
max_gradient_norm = 0.5  # to be defined

LEARNING_RATE = 0.02
learning_rate_decay_factor = 0.95
l1_scale = 0.000

num_examples = len(X_train)
max_steps = int(epochs * (num_examples / batch_size))
learning_rate_adjustments = 10
adjust_learning_rate_after_steps = max_steps / learning_rate_adjustments

# some tf varioables
tf_keep_probabiltiy = tf.constant(dropout_keep_probability)
tf_global_step = tf.Variable(0, trainable=False)

# clustering hierarchy
examples_in_hierarchy = 50
color_threshhold = 0.2

# threshold for vmeasure, homogenity etc.
h_threshold = 0.004

# save hyperparameters
hyperparams = {
    "state_size": state_size,
    "num_examples_to_visualize": num_examples_to_visualize,
    "dropout_keep_probability": dropout_keep_probability,
    "num_lstm_layers": num_lstm_layers,
    "learning_rate_decay_factor": learning_rate_decay_factor,
    "batch_size": batch_size,
    "dtype": str(DTYPE),
    "num_samples": num_samples,
    "max_gradient_norm": max_gradient_norm,
    "learning_rate": LEARNING_RATE,
    "epochs": epochs,
    "num_examples": num_examples,
    "max_steps": max_steps,
    "examples_in_hierarchy": examples_in_hierarchy,
    "color_threshhold": color_threshhold,
    "h_threshold": h_threshold,
}

save_to_json(hyperparams, hyper_parameter_file)

# Graph

learning_rate = tf.train.exponential_decay(
    learning_rate=LEARNING_RATE,
    global_step=tf_global_step,  # current learning step
    decay_steps=adjust_learning_rate_after_steps,  # how many steps to train after decaying learning rate
    decay_rate=learning_rate_decay_factor,
    staircase=True)

# Input, Output and Target of the graph

# inputs, outputs
x_e = tf.placeholder(tf.int32, [batch_size, None])  # encoder inputs loglines [batch_size, num_steps]
x_d = tf.placeholder(tf.int32, [batch_size, None])  # decoder inputs
y_d = tf.placeholder(tf.int32, [batch_size, None])  # reversed loglines [batch_size, num_steps]

visualization_embeddings = tf.Variable(np.zeros([num_examples_to_visualize, state_size]), trainable=False,
                                       name="VisualizationEmbeddings", dtype=DTYPE)

word_embeddings = tf.get_variable('word_embeddings', [vocabulary_size, state_size],
                                  dtype=DTYPE)  # each row is a dense vector for each word.

# Encoder Inputs
encoder_inputs = tf.nn.embedding_lookup(word_embeddings, x_e)  # [batch_size, max_time, embedding_size]
encoder_sequence_lengths = tf.placeholder(tf.int32, [batch_size])  # [batch_size]

# Decoder Inputs
decoder_inputs = tf.nn.embedding_lookup(word_embeddings,
                                        x_d)  # need to be defined  [batch_size, max_time, embedding_size]
decoder_sequence_lengths = tf.placeholder(tf.int32, [batch_size])  # [batch_size]

# Decoder Labels
decoder_labels = y_d  # this are our target words

# LSTM, Dropout Wrapper, MultiRNNCell
from library.core_rnn_cell_impl import \
    DropoutWrapper as DtypeDropoutWrapper  # import v1.1.0 dropout wrapper to support setting DTYPE to half-precision

# Define cells
with tf.variable_scope("encoder_scope") as encoder_scope:
    cell_fw = contrib_rnn.LSTMCell(num_units=state_size, state_is_tuple=True)
    cell_fw = DtypeDropoutWrapper(cell=cell_fw, output_keep_prob=tf_keep_probabiltiy, dtype=DTYPE)
    cell_fw = contrib_rnn.MultiRNNCell(cells=[cell_fw] * num_lstm_layers, state_is_tuple=True)

    cell_bw = contrib_rnn.LSTMCell(num_units=state_size, state_is_tuple=True)
    cell_bw = DtypeDropoutWrapper(cell=cell_bw, output_keep_prob=tf_keep_probabiltiy, dtype=DTYPE)
    cell_bw = contrib_rnn.MultiRNNCell(cells=[cell_bw] * num_lstm_layers, state_is_tuple=True)

    encoder_cell_fw = cell_fw
    encoder_cell_bw = cell_bw

    encoder_outputs, (output_state_fw, output_state_bw) = tf.nn.bidirectional_dynamic_rnn(
        cell_fw=encoder_cell_fw,
        cell_bw=encoder_cell_bw,
        dtype=DTYPE,
        sequence_length=encoder_sequence_lengths,
        inputs=encoder_inputs,
    )

    last_encoder_state = tuple(map(

        lambda fw_state, bw_state: tf.contrib.rnn.LSTMStateTuple(

            c=tf.concat((fw_state.c, bw_state.c), 1,
                        name="bidirectional_concat_c"),

            h=tf.concat((fw_state.h, bw_state.h), 1,
                        name="bidirectional_concat_h")),

        output_state_fw, output_state_bw))

# Dynamic RNN decoder

with tf.variable_scope("decoder_scope") as decoder_scope:
    # output projection
    # we need to specify output projection manually, because sampled softmax needs to have access to the the projection matrix
    output_projection_w_t = tf.get_variable("output_projection_w", [vocabulary_size, state_size*2], dtype=DTYPE)
    output_projection_w = tf.transpose(output_projection_w_t)
    output_projection_b = tf.get_variable("output_projection_b", [vocabulary_size], dtype=DTYPE)

    # define decoder cell
    decoder_cell = tf.contrib.rnn.LSTMCell(num_units=state_size*2)
    decoder_cell = DtypeDropoutWrapper(cell=decoder_cell, output_keep_prob=tf_keep_probabiltiy, dtype=DTYPE)
    decoder_cell = contrib_rnn.MultiRNNCell(cells=[decoder_cell] * num_lstm_layers, state_is_tuple=True)
    # decoder_cell = contrib_rnn.OutputProjectionWrapper(decoder_cell, output_size=vocabulary_size ) # rnn output: [batch_size, max_time, vocabulary_size ]

    # define decoder train netowrk
    decoder_outputs_tr, _, _ = dynamic_rnn_decoder(  # decoder outputs, final hidden state, final context state
        cell=decoder_cell,  # the cell function
        decoder_fn=simple_decoder_fn_train(last_encoder_state, name=None),
        inputs=decoder_inputs,  # [batch_size, max_time, embedding_size].
        sequence_length=decoder_sequence_lengths,  # length for sequence in the batch [batch_size]
        parallel_iterations=None,  # Tradeoff - time for memory
        swap_memory=False,
        time_major=False)

    # define decoder inference network
    decoder_scope.reuse_variables()

# Loss function


print("Decoder Output  Shape: [batch_size=%s, max_timesteps=%s, vocabulary_size=%s]" % tuple(decoder_outputs_tr.shape))
# from library.nn_impl import sampled_softmax_loss as dtype_sampled_softmax_loss # imported cutom loss function to support dtype  as a parameter => half precision

# reshape outputs of decoders to [ batch_size * max_time , vocabulary_size ]
decoder_forward_outputs = tf.reshape(decoder_outputs_tr, [-1, state_size*2])
decoder_target_labels = tf.reshape(decoder_labels, [-1, 1])  # =>  [ batch_size * max_time ]  sequence of correc tlabel

sampled_softmax_losses = tf.nn.sampled_softmax_loss(
    weights=output_projection_w_t,  # [num_classes, state_size]
    biases=output_projection_b,  # [num_classes]
    inputs=decoder_forward_outputs,
    # inputs: A `Tensor` of shape `[batch_size, state_size]`.  The forward activations of the input network.
    labels=decoder_target_labels,
    num_sampled=num_samples,
    num_classes=vocabulary_size,
    num_true=1,
)
total_loss_op = tf.reduce_mean(sampled_softmax_losses)

# Regularization

l1_regularizer = tf.contrib.layers.l1_regularizer(
    scale=l1_scale, scope=None
)
weights = tf.trainable_variables()
regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, [word_embeddings])

regularized_loss = total_loss_op + regularization_penalty

# Trainings step

# get gradients for all trainable parameters with respect to our loss funciton
params = tf.trainable_variables()
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
gradients = tf.gradients(regularized_loss, params)

# apply gradient clip
clipped_gradients, gradient_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)

# Update operation
training_step = optimizer.apply_gradients(zip(clipped_gradients, params),
                                          global_step=tf_global_step)  # learing rate decay is calulated based on this step

# Training

# load single example from processed file
filename_queue = tf.train.string_input_producer([processed_datafile], num_epochs=None)  # num_epochs=None
reader = tf.TextLineReader()
example_id, single_example_logline = reader.read(filename_queue)

# define how to parse single example (see 1.3)
split_example = tf.string_split([single_example_logline], delimiter="|")
split_example_dense = tf.sparse_tensor_to_dense(split_example, default_value='', validate_indices=True, name=None)
split_example_dense = split_example_dense[0]

# split_example_dense[0] is signature id
# sequence length
sequence_length = tf.string_to_number(split_example_dense[1], out_type=tf.int32)
# split encoder inputs
enc_split = tf.string_split([split_example_dense[2]], delimiter=",")
enc_split_dense = tf.sparse_tensor_to_dense(enc_split, default_value='', validate_indices=True, name=None)
x_e_single = tf.string_to_number(enc_split_dense, out_type=tf.int32)
# split decoder inputs
dec_split = tf.string_split([split_example_dense[3]], delimiter=",")
dec_split_dense = tf.sparse_tensor_to_dense(dec_split, default_value='', validate_indices=True, name=None)
x_d_single = tf.string_to_number(dec_split_dense, out_type=tf.int32)
# split decoder targets
tar_split = tf.string_split([split_example_dense[4]], delimiter=",")
tar_split_dense = tf.sparse_tensor_to_dense(tar_split, default_value='', validate_indices=True, name=None)
y_single = tf.string_to_number(tar_split_dense, out_type=tf.int32)

# Batch the variable length tensor with dynamic padding
fetch_trainings_batch = tf.train.batch(
    tensors=[
        x_e_single[0],  # single encoder input line
        x_d_single[0],  # single decoder input line
        y_single[0],  # single decoder target line
        sequence_length,  # sequence length of this example
    ],
    batch_size=batch_size,
    dynamic_pad=True,
    name="trainings_batch",
    allow_smaller_final_batch=True
)


# Session Setup

def get_batch_dict(session):
    # get trainingsbatch from input queue
    batch = session.run([fetch_trainings_batch], feed_dict=None)
    batch = batch[0]

    # assign arrays to dictionary
    batch_dict = {
        x_e: batch[0],  # encoder inputs
        x_d: batch[1],  # decoder inputs
        y_d: batch[2],  # decoder targets
        encoder_sequence_lengths: batch[3],
        decoder_sequence_lengths: batch[3]
    }
    return batch_dict


# Trainings Loop

if learn_model:
    current_epoch = 0
    save_checkpoint_after_each_step = int(max_steps / 10)
    print_loss_after_steps = int(max_steps / 100)

    queue_capacity = 2 * batch_size

    # Saver
    saver = tf.train.Saver(tf.global_variables())

    # Summaries
    tf.summary.scalar("total_loss", tf.cast(total_loss_op, DTYPE))  # summary for accuracy
    tf.summary.scalar("regularized_loss", tf.cast(regularized_loss, DTYPE))  # summary for accuracy
    # Start session
    session = tf.Session()

    # Check random seed for reproducibility
    control_random_number = session.run(tf.random_normal([1]))
    print("Control Random Number: %0.5f" % control_random_number)  # should change if you modify the graph

    all_summaries = tf.summary.merge_all()
    # summary_writer = tf.summary.FileWriter(graph_dir, graph=session.graph)

    session.run([
        tf.local_variables_initializer(),
        tf.global_variables_initializer(),
    ])

s = time.time()  # start time
if learn_model:

    word_embeddings_before = session.run(word_embeddings)
    import collections

    batch_times = collections.deque()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=session, coord=coord)

    for current_step in range(1, max_steps + 1):  # start from 1 .. max_steps+1 to execute max steps

        try:
            step_s = time.time()
            # increase step counter
            session.run(tf_global_step.assign(current_step))

            # get next batch
            batch_dict = get_batch_dict(session)
            # execute actions
            results = session.run([
                total_loss_op,  # calculate training loss
                regularized_loss,  #
                training_step,  # calculate gradients, update gradients
                all_summaries,  # compile summaries and write to graph dir,
                learning_rate,
                # get gradients
            ], feed_dict=batch_dict)

            # summary_writer.add_summary(results[3], current_step)
            batch_times.append((time.time() - step_s))

        except Exception as e:
            # Report exceptions to the coordinator.
            print("Aborting trainingsloop :%s" % str(e.message))
            coord.request_stop(e)
            break

        if current_step % print_loss_after_steps == 0:
            print(
                "Epoch {epoch:02d}, Step {current_step:05d}/{max_steps:05d}, Current Learning rate: {learning_rate:0.4f},  Loss: {loss:0.4f} Regularized Loss: {regularized_loss:0.4f}".format(
                    epoch=int(current_step / (max_steps / epochs) + 1),
                    current_step=current_step, max_steps=max_steps,
                    learning_rate=results[4],
                    loss=results[0], regularized_loss=results[1]
                )  # end format
            )  # end print
            avg_batch_time = sum(batch_times) / len(batch_times)
            total_time_in_s = avg_batch_time * max_steps
            print("Average step time: %0.2fs, Estimated total duration: ~%0.2f min (~ %0.2f h) " % (
                avg_batch_time, total_time_in_s / 60.0, total_time_in_s / 3600.0))

    coord.request_stop()
    word_embeddings_after = session.run(word_embeddings)
    e = time.time()
    print("Learning took {sec:0.2f} seconds".format(sec=(e - s)))

# Embed test log lines 2000

embedded_csv = join_path(result_dir, experiment_dir, "embedded_lines.csv")
embedded_loglines = np.ndarray(shape=[num_examples_to_embed, state_size*2])

if not os.path.exists(embedded_csv):
    # load
    test_examples = list(open(test_data_processed, "r"))
    print(len(test_examples))
    test_examples = [t.split("|") for t in test_examples]
    print("Loaded training data")

    s = time.time()
    count = 0
    train_batches = list(range(0, num_examples_to_embed, batch_size))
    print("Started embedding %i batches of %i log lines" % (len(train_batches), num_examples_to_embed))
    for i, ex_id in enumerate(train_batches):
        x_e_batch = []
        esl_batch = []

        incomplete_batch = False  # for last batch
        incomplete_length = 0
        # get batch, write metadata
        for b_id, example in enumerate(test_examples[ex_id:ex_id + batch_size]):
            count += 1
            esl_batch.append(int(example[1]))  # sequence length
            x_e_batch.append(np.array(example[2].split(","), dtype=np.integer))  # encoder input

        if len(x_e_batch) < batch_size:  # last_batch, so need to zero add id
            incomplete_batch = True
            incomplete_length = len(x_e_batch)

            for _ in range(batch_size - len(x_e_batch)):
                esl_batch.append(0)
                x_e_batch.append(np.zeros([len(x_e_batch[0])]))

        # pad batch
        max_seq_len = np.amax(esl_batch)  # maximum encoder length
        padded_x_e_batch = []
        for x_es in x_e_batch:
            padded_x_e = np.pad(x_es, (0, max_seq_len - len(x_es)), 'constant', constant_values=0)
            padded_x_e_batch.append(padded_x_e)

            # assign arrays to dictionary
        batch_dict = {
            x_e: np.array(padded_x_e_batch),  # encoder inputs  # [batch_size, max_sequence_length, state_size ]
            encoder_sequence_lengths: np.array(esl_batch)  # encoder sequence length [batch_size,1]
        }

        # execute actions
        results = session.run([
            last_encoder_state,  # get encoded trainings samples
        ], feed_dict=batch_dict)

        # story in temporary array
        if incomplete_batch:
            embedded_loglines[ex_id:ex_id + incomplete_length, ] = results[0][0].c[0:incomplete_length]
        else:
            embedded_loglines[ex_id:ex_id + batch_size, ] = results[0][0].c  # copy c hidden state to tmp_ary

    print("Embedded %i lines in ~%0.2f min" % (num_examples_to_embed, ((time.time() - s) / 60.0)))
else:
    embedded_loglines = np.genfromtxt(embedded_csv, delimiter=";")
    print("Loaded embedded lines to file: %s " % embedded_csv)
    print(embedded_loglines.shape)

X = embedded_loglines
print(X)

np.savetxt(result_dir + '/' + experiment_dir + "/" + EXPERIMENT_ID + "_" + experiment_sub_dir + "_" + "embeddings.csv",
           X, delimiter=",")
