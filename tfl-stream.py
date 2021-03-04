import sounddevice as sd
import numpy as np
import timeit
import tensorflow.compat.v1 as tf


# Parameters
debug_time = 0
debug_acc = 0
word_threshold = 10.0
rec_duration = 0.020
sample_rate = 16000
num_channels = 1

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models2/crnn_state/quantize_opt_for_size_tflite_stream_state_external/stream_state_external.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

last_argmax = 0
out_max = 0
hit_tensor = []
inputs = []
for s in range(len(input_details)):
  inputs.append(np.zeros(input_details[s]['shape'], dtype=np.float32))
    
def sd_callback(rec, frames, time, status):

    global last_argmax
    global out_max
    global hit_tensor
    global inputs
    start = timeit.default_timer()
    
    # Notify if errors
    if status:
        print('Error:', status)
    
    rec = np.reshape(rec, (1, 320))
    
    # Make prediction from model
    interpreter.set_tensor(input_details[0]['index'], rec.astype(np.float32))
    # set input states (index 1...)
    for s in range(1, len(input_details)):
      interpreter.set_tensor(input_details[s]['index'], inputs[s])
  
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    # get output states and set it back to input states
    # which will be fed in the next inference cycle
    for s in range(1, len(input_details)):
      # The function `get_tensor()` returns a copy of the tensor data.
      # Use `tensor()` in order to get a pointer to the tensor.
      inputs[s] = interpreter.get_tensor(output_details[s]['index'])
      
    out_tflite_argmax = np.argmax(output_data)
    if last_argmax == out_tflite_argmax:
      if output_data[0][out_tflite_argmax] > out_max:
        out_max = output_data[0][out_tflite_argmax]
        hit_tensor = output_data
    else:
      print(last_argmax, out_max, hit_tensor)
      out_max = 0
    
    last_argmax = out_tflite_argmax
    
    #if out_tflite_argmax == 2:
        #print('raspberry')
        #print(output_data[0][2])
        

    if debug_acc:
        print(out_tflite_argmax)
    
    if debug_time:
        print(timeit.default_timer() - start)

# Start streaming from microphone
with sd.InputStream(channels=num_channels,
                    samplerate=sample_rate,
                    blocksize=int(sample_rate * rec_duration),
                    callback=sd_callback):
    while True:
        pass
