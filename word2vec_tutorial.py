import os
import sys
import math
import random
import urllib
import zipfile
import argparse
import collections

import numpy as np
import tensorflow as tf

# Use tempfile module allows you to write to a tempfile which is closed
# when the script is finished
from tempfile import gettempdir

# provide folder path as an argument with '--log_dir' to save tensorboard summaries
current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
parser = argparse.ArgumentParser()
parser.add_argument(
    '--log_dir',
    type=str,
    default=os.path.join(current_path, 'log'),
    help='The log directory for Tensorboard summaries.')
# FLAGS is the optional --log_dir input to the script. If this is not given
# it will default to the current directory and the next step will actually
# create that log directory
FLAGS, unparsed = parser.parse_known_args()

# Create directory for Tensorboard variables if there is not
if not os.path.exists(FLAGS.log_dir):
    os.makedirs(FLAGS.log_dir)

# download data
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    '''Download a file if not present, and make sure its the right size'''
    local_filename = os.path.join(gettempdir(), filename)
    if not os.path.exists(local_filename):
        print(url + filename)
        # copies network object denoted by URL to local_filename. Will not
        # be copies unless local_filename is true. Here we are copying it
        # to a temporary directory.
        local_filename, _ = urllib.request.urlretrieve(url + filename, local_filename)

    statinfo = os.stat(local_filename)  # Get stats on file created (i.e. its size)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verity ' + local_filename +
                        '. Can you get to it with a browser?')
    return local_filename


filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
    '''Etract the first file enclosed in a zip file as a list of words'''
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()  # making string compatibility between Python 2 & 3
    return data


vocabulary = read_data(filename)
print('Data size', len(vocabulary))

# Build dictionary and replace rare words with UNK (unknown) token
vocabulary_size = 50000
print(vocabulary[0:5])


def build_dataset(words, n_words):
    '''Process raw inputs into a dataset'''
    count = [['UNK', -1]]
    # Count the most common words in the list provided.
    # '.extend' is similar to appending a list with another iterable (i.e. another list)
    # now list has [['UNK', -1], (word1, freq1), (word2, freq2)...] '''
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
        # As the most common words in count are entered first, they will receive
        # a higher value (i.e. closer to 1) in the dictionary, allowing us to
        # use this to select for these common words later
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)  # return 0 if word not in dictionary
        if index == 0:  # i.e. dictionary['UNK'] or infrequent words
            unk_count += 1
        data.append(index)  # Create a list of all the indices
    count[0][1] = unk_count  # Change UNK count from 0 to the number of unknown (UNK) words in the dataset

    # Swap the keys and values around in the dictionary. This enables us to select
    # a word by using its rank of frequency as the dictionary key.
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


# Filling 4 global variables:
# data - list of codes (integers from 0 to vobulary_size-1)
# count - map of words(strings) to count of occurences
# dictionary - map of words (strings) to their codes (integers)
# reverse_dictionary - maps codes (integers) to their words
data, count, dictionary, reverse_dictionary = build_dataset(
    vocabulary, vocabulary_size)
del vocabulary  # reduces memory
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

data_index = 0

# *** FUNCTION TO GENERATE TRAINING BATCH FOR THE SKIP-GRAM MODEL *** #

'''
To better understand what this function below is doing, pretend we fed "the quick
brown fox jumped over the lazy dog". The skip_window is the number around the word
of interest that you want to include as your context word. num_skips is the
number of words you want to sample as context words. So for num_skips = 2,
skip_window = 1, this is like saying if we focus on the word "quick", we will
want to select "the" and "fox" from around it.
'''


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    # using assert for debugging. If batch_size not perfectly divisible
    # by num_skips then an error will be raised.
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # Create a span of words which covers your target word and the
    # context words around it
    span = 2 * skip_window + 1
    # deque is like a list but allows appending and popping from either side
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    # Add to buffer data from some index to the span length, defined by skip_window
    buffer.extend(data[data_index:data_index + span])
    data_index += span  # moves data_index along by 'span' units
    for i in range(batch_size // num_skips):
        # avoids introducing number for w when at skip_window. These
        # context_words are then the words _around_ the word of interest.
        # i.e. we avoid selection "quick" and select "the" and "brown"
        context_words = [w for w in range(span) if w != skip_window]
        # samples num_skips items randomly from population context_words.
        # This is why num_skips has to be less than 2 * skip_window. Otherwise
        # it will sample more than the span elements that are in context_words.
        # For num_skip = 2 we will grab both "the" and "brown", although we
        # could have set num_skip lower and only selected one of these words
        words_to_use = random.sample(context_words, num_skips)

        for j, context_word in enumerate(words_to_use):
            # iterates through j num_skips times (because this is the number
            # of samples in words_to_use). So we will get "the" and "brown"
            # for the target word of "quick"
            batch[i * num_skips + j] = buffer[skip_window]  # selects word of interest
            labels[i * num_skips + j, 0] = buffer[context_word]  # selects a word from _around_ the word of interest

        # now we have grabbed the surrounding words from for our word of
        # interest and we move onto the next word. In the else statement
        # we move along one word to the right. By using the deque buffer,
        # it will get appended to the end and remove the left most word
        # from before (i.e. "the")
        if data_index == len(data):
            # If the index has reached the end of the data then loop around
            # and begin putting the early values of the data in again
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])  # using append as we're just adding a single value.
            data_index += 1
    # Backtrack to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
for i in range(8):
    print(
        batch[i],
        reverse_dictionary[batch[i]],
        '->', labels[i, 0], reverse_dictionary[labels[i, 0]]
    )  # should print out the labels selected for what word is in batch[i]

# *** BUILD AND TRAIN SKIP-GRAM MODEL *** #

batch_size = 128
embedding_size = 128  # Dimension of the embeding vector
skip_window = 1  # How many words to consider left and right
num_skips = 2  # How many times to reuse an input to generate a label
num_sampled = 64  # Number of negative examples to sample

# Pick a random validation set to sample nearest neighbours.
# We limit the validation samples to the words that have a
# low numeric ID, which by construction are also the most frequent
# These three variables are used only for displaying model
# accuracy and won't affect calculation.

valid_size = 16  # Random set of words to evaluate similarity on
valid_window = 100  # Only pick dev samples in the head of the distribution
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# Create random embeddings to begin with for each word in vocabulary, with a dimension of embedding_size
embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
embed = tf.nn.embedding_lookup(embeddings, train_inputs)  # embedding_lookup simply gets the embeding vector for each row (word in vocab) which is given by train_inputs

# Construct the variable for the NCE loss
nce_weights = tf.Variable(
    tf.truncated_normal(
        [vocabulary_size, embedding_size],
        stddev=1.0 / math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Compute the average NCE loss for the batch.
# nce_loss automatically draws a new sample of the negative labels each
# time we evaluate the loss.name_scope
# A good overview is here: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
loss = tf.reduce_mean(
    tf.nn.nce_loss(
        weights=nce_weights,
        biases=nce_biases,
        labels=train_labels,
        inputs=embed,
        num_sampled=num_sampled,
        num_classes=vocabulary_size))

# Add the loss value as a scalar to summary
tf.summary.scalar('loss', loss)

optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

# Compute the cosine similarity between mnibatch examples and all embeddings
# Below calculates the absolute sum for each of the dimensions in the embedding vector
# This is then used to normalise the word embeddings
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)

# All this most frequent words "the" "and" "it" should appear in the same
# contexts and be very similar. Therefore we can use these to see how well
# our model is doing
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Add variable initializer
init = tf.global_variables_initializer()

# BEGIN TRAINING
num_steps = 100001

with tf.Session() as session:
    # We must initialise all variables before we use them
    init.run()
    print('Initialized')

    average_loss = 0
    for step in range(num_steps):
        # Use our function from before to generate small batches of words
        # to train our model on
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # We perform one update step by evaluating the optimizer op
        # (including it in the list of retrned values for session.run())
        # Also, evaluate the merged op to get all summaries from the returned "summary" variable
        _, loss_val = session.run(
            [optimizer, loss],
            feed_dict=feed_dict)
        average_loss += loss_val  # keep tally of loss calculated

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000
            # Use tally of loss calculated to work out the average loss
            # for every 2000 steps
            print(f'Average losss at step {step}: {average_loss}')
            average_loss = 0

        # Every 10000 steps show progress of the training by demonstrating
        # which words are close to the validation words set previously
        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neigbours

                # Similar word vectors will have a greater magnitude when
                # multiplied together than dissimilar ones. Taking the
                # negative of this multiplication means the most similar
                # word vectors are more negative. Then performing an argsort
                # allows us to find the indices of these most negative
                # (i.e. most similar) vectors and select them
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = f'Nearest to {valid_word}:'
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = f'{log_str} {close_word}'
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
