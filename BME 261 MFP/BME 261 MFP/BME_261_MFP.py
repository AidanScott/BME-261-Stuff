import sounddevice as sd                                            #library with functions for reading audio data from a microphone to a numpy array
import soundfile as sf                                              #library with functions for reading, writing and otherwise manipulating audio files
import scipy as sp                                                  #library with various math functions and computation/analysis tools                                             
import numpy as np                                                  #library with functions for math, particularly relating to arrays and array manipulation
import matplotlib.pyplot as plt                                     #library with functions for visual plotting of data

import sys                                                          #generically useful library for low-level i/o
import queue                                                        #used in the callback function for InputStream

                                                                    #sounddevice would be replaced by the functionally similar pyaudio in a high-fidelity prototype capable of running on mobile devices through Kivy


def record(filename):                                               #function for taking recordings; kind of complicated, will break down w/ further comments

    q = queue.Queue()

    def callback(data,frames,time,status):                          #sub-function called by InputStream; essentially defines where it's putting the raw audio data it collects
        if status:
            print(status, file=sys.stderr)
        q.put(data.copy())

    try:                                                            #the important part. Probably doesn't work the way I want it to atm? we'll get there though. Step up from the version with parsers at least
        with sf.SoundFile(filename, mode='x', samplerate=44100, channels=1) as file:
            with sd.InputStream(samplerate=44100,device=mic,channels=1,callback=callback):
                print('#'*80)
                print('press Ctrl+C to stopp recording')
                print('#'*80)
                while True:
                    file.write(q.get())
    except KeyboardInterrupt:
        print('\nRecording Finished: ' + filename)
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))
    







