import sounddevice as sd                                            #library with functions for reading audio data from a microphone to a numpy array
import soundfile as sf                                              #library with functions for reading, writing and otherwise manipulating audio files
import scipy as sp                                                  #library with various math functions and computation/analysis tools     
import scipy.signal as sig
import numpy as np                                                  #library with functions for math, particularly relating to arrays and array manipulation
import matplotlib.pyplot as plt                                     #library with functions for visual plotting of data

import sys                                                          #generically useful library for low-level i/o
import queue                                                        #used in the callback function for InputStream
import threading                                                    #library for parralelism and multithreading; not strictly necessary for MFP but would be for anything higher fidelity, and something I'd like to learn better, so might as well

                                                                    #sounddevice would be replaced by the functionally similar pyaudio in a high-fidelity prototype capable of running on mobile devices through Kivy


is_recording = True                                                 #global bool that records current recording state; would be bound to a button in the UI in a higher fidelity prorotype

def threaded_recording(filename,q):                                 #function for recording; intended to be run in parallel to allow for arbitrary recording duration

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
                print('Press enter to stop recording')
                print('#'*80)
                while is_recording:
                    file.write(q.get())
    except KeyboardInterrupt:
        print('\nRecording ended unexpectedly, recording may be incomplete: ' + filename)
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))


def record(filename):                                               #function for taking recordings; kind of complicated, will break down w/ further comments
    
    q = queue.Queue()                                               #queue object that temporarily stores audio data as it comes in

                                                                    #thread object that implements the parallel recording function
    recording = threading.Thread(target=threaded_recording, args=(filename,q,))

    recording.start()                                               #recording starts here

    input()                                                         #main execution blocked by user input

    global is_recording                                             #after user input recieved, the while loop in the thread is set to false, and it exits

    is_recording = False                                                                 
    
                                                                    #function that performs first processing steps
                                                                    #converts audio file to numpy array, filters out irrelevant frequencies
                                                                    #returns as numpy arrays: original signal, filtered signal, fft transform
                                                                    #returns sample rate as int
def preprocessing(filename):                                       

    signal, sr = sf.read(filename)                                  #reads audio file to np.array (signal), and returns sampling rate as an int (sr) 

    if(signal.ndim > 1):                                            #checks the number of channels in the audio file
        signal = np.add(signal[:,0],signal[:,1])                    #if the file is stereo, the array is converted from 2D to 1D

                                                                    #defines a 10th order band-pass filter between 20 and 1000Hz; excluding sounds outside the range of heart signals
    band_pass_filter = sig.butter(10,[20,1000],'bp',output='sos',fs=sr)

    filtered_signal = sig.sosfilt(band_pass_filter, signal)         #applies the filter to the signal

    filtered_signal = filtered_signal/np.max(filtered_signal)       #normalizes filtered signal to magnitude 1

    signal_fft = sp.fft.fft(filtered_signal)                        #returns fast fourier transform of the signal

    return signal, sr, filtered_signal, signal_fft


def display_signal(signal, sr, id, mode):                           #generates a graphical display of the input signal, depending on the designated mode
                                                                    #displays: time-domain representation 
                                                                    #frequency-domain representation
                                                                    #spectrogram
                                                                    #inputs: signal; 1darray
                                                                    #sr (sample rate); int
                                                                    #id; str, used in chart titles
                                                                    #mode; str, designates display type
                                                                    #time: expects filtered or input signal
                                                                    #freq: expects filtered fft
                                                                    #spec: expects filtered signal

    if(mode == 'time'):                                             #plots and displays signal amplitude vs time, pretty self-explanitory

        t = np.linspace(0.0,signal.size/sr,signal.size)

        plt.plot(t,signal)
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.title('Time Domain: ' + id)
        plt.show()
        plt.clf()

    elif(mode == 'freq'):                                           #plots and displays fourier transform power vs frequency

        signal_freq = sp.fft.fftfreq(signal.size, d=(1/sr))         #returns an array containing the frequency range of the fourier transform (-sr/2 to sr/2, I believe, so it could be manually defined, but this is hopefully less prone to error.)

                                                                    #notes on the actual plotting:
                                                                    #area being plotted is restricted to positive between 0 and 1000, to make the plot actually readable
        plt.plot(signal_freq[0:signal.size*1000//sr], 2.0/signal.size*np.abs(signal[0:signal.size*1000//sr]))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power')
        plt.grid()
        plt.title('Frequency Domain: ' + id)
        plt.show()
        plt.clf()

    elif(mode == 'spec'):                                           #generates and displays signal spectrogram

        f, t, s = sig.spectrogram(signal,sr)                        #scipy spectrogram function

        plt.pcolormesh(t,f,s, shading='gouraud')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [s]')
        plt.axis([0,signal.size/sr,0,1000])                         #restrics area plotted to 0,1000
        plt.title('Spectrogram: ' + id)
        plt.show()
        plt.clf()
    

#record('testing.wav')                                              #recording testing  

processed_signal = preprocessing('Apex16(noisy).au')                #processing testing
                                                                    #file is a heart sound (Apex16.au) overlaid with three sin waves, one at 10 hz, one at ~2000hz and one at ~1750hz

                                                                    #display testing
display_signal(processed_signal[0],processed_signal[1],'Input','time')
display_signal(processed_signal[2],processed_signal[1],'Filtered','time')
display_signal(processed_signal[3],processed_signal[1],'FFT of Filtered','freq')
display_signal(processed_signal[2],processed_signal[1],'Filtered','spec')

    







