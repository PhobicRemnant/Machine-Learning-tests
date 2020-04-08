# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 16:29:46 2020

Make a SoulsBorne series boss name generator, excluding Sekiro boss's names.

This code is made out of the examples in the page of TensorFlow for text
generation with RNNs: https://www.tensorflow.org/tutorials/text/text_generation

The purpose of this code is mainly educational for my own dark elusive purposes
in serving the machine.

Some differences are to be noted as in the output and the size of the data set,
the SoulsBorne series has only 159 bosses with aproximately 18 characters per
name.



@author: The Great Cephalopod
"""

import tensorflow as tf

import numpy as np
import os
import time

def split_input_target(chunk):
    """
    This function is applied to each batch using the map method
    """
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    """
    Function to ease the building of a RNN model for training
    """
    model = tf.keras.Sequential([
    
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                       kernel_initializer='glorot_uniform'),
    tf.keras.layers.GRU( int(rnn_units/2),
                         return_sequences=True,
                         stateful=True,
                         recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
    ])
  
    return model

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 20

  # Converting our start string to numbers (vectorizing)
  input_eval = [char_to_int[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 0.5

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(int_to_char[predicted_id])

  return (start_string + ''.join(text_generated))

"""
1.- Load the names from "SoulsBorne_bossNames.txt" 
"""

data = open('SoulsBorne_bossNames.txt','r').read()
data = data.lower() # Try commenting this line to experiment with RNN training
chars = list(set(data))
chars = sorted(chars)
data_size, vocab_size = len(data), len(chars)

data_names = data.split('\n')

avrg_name_size = 0
for i in range(len(data_names)):
    avrg_name_size = avrg_name_size + len(data_names[i])
avrg_name_size = avrg_name_size/len(data_names)


print("There are %d characters from a total %d in the data" % (vocab_size, data_size))

#Define a char to int conversion to be able to deal with letters 
char_to_int = { ch:i for i,ch in enumerate(chars) }
#int_to_char = { i:ch for i,ch in enumerate(chars) }
int_to_char = np.array(chars)

data_int = np.array( [char_to_int[c] for c in data ] )

print ('{} ---- characters mapped to int ---- > {}'.format(repr(data[:13]), data_int[:13]))

"""
2.- Create training examples and targets
"""

# Maximum sequence lenght for an input
seq_length = 20
examples_per_epoch = len(data)//(seq_length+1)

# Create training data examples  
char_dataset = tf.data.Dataset.from_tensor_slices(data_int)

for i in char_dataset.take(10):
  print(int_to_char[i.numpy()])

# Use the batch method for selecting sequence
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

for item in sequences.take(5):
  print(repr(''.join(int_to_char[item.numpy()])))
  
dataset = sequences.map(split_input_target)

# Print the first examples' inputs and target values
for input_example, target_example in  dataset.take(1):
  print ('Input data: ', repr(''.join(int_to_char[input_example.numpy()])))
  print ('Target data:', repr(''.join(int_to_char[target_example.numpy()])))

# Show inputs and targets at time stamp
for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(int_to_char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(int_to_char[target_idx])))

# Configure batch size
BATCH_SIZE = 16

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).

BUFFER_SIZE = 25

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


"""
3.- Build your RNN model
"""

model = tf.keras.Sequential()

# Length of the vocabulary in chars
vocab_size = len(chars)

# The embedding dimension
embedding_dim = 128
# Number of RNN units
rnn_units = 564

# Call build_model function to create the RNN
model = build_model(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)

"""
Test the model for the first batch
"""

# Check the shape of the output
for input_example_batch, target_example_batch in dataset.take(1):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

# Show model summary
model.summary()

# Try the model for the first batch 
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())


"""
4.- Train the model
"""

# Configure the training process for the model
model.compile(optimizer='adam', loss=loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

# Set the numper of ephocs to create a history for the model
EPOCHS= 100
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])

"""
5.- Generate text with model
"""

# Declare the variable checkpoint dir with the weights of the trained RNN
tf.train.latest_checkpoint(checkpoint_dir)

# Remember to change the batch size to 1 since the training stage is over
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

# Load the weights of the checkpoint
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

# Show the trained model to compare with trained
model.summary()


print(generate_text(model, start_string="keeper"))


