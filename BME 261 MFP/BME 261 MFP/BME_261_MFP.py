import sounddevice as sd                                            #library with functions for reading audio data from a microphone to a numpy array
import soundfile as sf                                              #library with functions for reading, writing and otherwise manipulating audio files
import scipy as sp                                                  #library with various math functions and computation/analysis tools                                             
import numpy as np                                                  #library with functions for math, particularly relating to arrays and array manipulation
import matplotlib.pyplot as plt                                     #library with functions for visual plotting of data

import sys                                                          #generically useful library for low-level i/o
import queue                                                        #used in the callback function for InputStream
import threading                                                    #library for parralelism and multithreading; not strictly necessary for MFP but would be for anything higher fidelity, and something I'd like to learn better, so might as well

                                                                    #sounddevice would be replaced by the functionally similar pyaudio in a high-fidelity prototype capable of running on mobile devices through Kivy


is_recording = True                                                 #global bool that records current recording state; would be bound to a button in the UI in a higher fidelity prorotype

def threaded_recording(filename,q):

    def callback(data,frames,time,status):                          #sub-function called by InputStream; essentially defines where it's putting the raw audio data it collects
        if status:
            print(status, file=sys.stderr)
        q.put(data.copy())
                                                                    #where recording actually happens
                                                                    #soundfile opens a file in the program directory for recording
                                                                    #inputstream places audio data from the microphone into the queue as it recieves it
                                                                    #data from the queue is copied and written into the sound file
                                                                    #this repeats until the global flag variable is_recording becomes false, or an exception occurs
                                                                    #in normal operation, is_recording will become false upon user input
    try:                                                           
        with sf.SoundFile(filename, mode='x', samplerate=44100, channels=1) as file:
            with sd.InputStream(samplerate=44100,channels=1,callback=callback):
                print('#'*80)
                print('press enter to stop recording')
                print('#'*80)
                while is_recording:
                    file.write(q.get())
    except KeyboardInterrupt:
        print('\nRecording Finished: ' + filename)
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))


def record(filename):                                               #function for taking recordings; kind of complicated, will break down w/ further comments
    
    q = queue.Queue()                                               #queue object that temporarily stores audio data as it comes in

                                                                    #thread object that implements the parallel recording function
    recording = threading.Thread(target=threaded_recording, args=(filename,q,))

    recording.start()                                               #recording starts herer

    input()                                                         #main execution blocked by user input

    global is_recording                                             #after user input recieved, the while loop in the thread is set to false, and it exits

    is_recording = False

                                            
                                                                    #recording testing
    
record('testing.wav')                                                                 
    
    







