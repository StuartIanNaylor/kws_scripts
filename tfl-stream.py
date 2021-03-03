"""
Connect a resistor and LED to board pin 8 and run this script.
Whenever you say "stop", the LED should flash briefly
"""

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import timeit
import python_speech_features
import tensorflow as tf


# Parameters
debug_time = 0
debug_acc = 0
word_threshold = 0.5
rec_duration = 0.2
sample_rate = 16000
num_channels = 1

# Sliding window
window = np.zeros(16000)

window_pos = 0

z = 0
# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details[0]['shape'])

def sd_callback(rec, frames, time, status):

    
    global window_pos
    global z
    # Start timing for testing
    start = timeit.default_timer()
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    window_pos = window_pos + 1
    # Remove 2nd dimension from recording sample
    rec = np.squeeze(rec)
    # Save recording onto sliding window
    np.roll(window,-3200)
    window[12800:] = rec
    if window_pos == 5:
      write("1sec-" + str(z) + ".wav", 16000, window)
      z = z + 1
      window_pos = 0
      
    #print(window, tf.shape(window))
    # Padding for files with less than 16000 samples
    zero_padding = tf.zeros([16000] - tf.shape(window), dtype=tf.float32)

    # Concatenate audio with padding so that all audio clips will be of the 
    # same length
    waveform = tf.cast(window, tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=1024, frame_step=512)

    spectrogram = tf.abs(spectrogram)
  
    # Warp the linear scale spectrograms into the mel-scale.
    num_spectrogram_bins = spectrogram.shape[-1]
    lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz, upper_edge_hertz)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

    # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)

    # Compute MFCCs from log_mel_spectrograms and take the first 13.
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :13]
    mfccs = tf.reshape(mfccs, [1, 30, 13,  1])
    #print(mfccs.shape)
    # Make prediction from model
    interpreter.set_tensor(input_details[0]['index'], mfccs)
  
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
    val = output_data[0][0]
    
    if val > word_threshold:
        print('raspberry')
        

    if debug_acc:
        print(val)
    
    if debug_time:
        print(timeit.default_timer() - start)

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass
