import sounddevice as sd                                            #library with functions for reading audio data from a microphone to a numpy array
import soundfile as sf                                              #library with functions for reading, writing and otherwise manipulating audio files
import pydub as dub                                                 #additional audio file library; doesn't interface with the analysis tools we're using as cleanly, but necessary to cover common file types soundfile can't
import scipy as sp                                                  #library with various math functions and computation/analysis tools     
import scipy.signal as sig
import numpy as np                                                  #library with functions for math, particularly relating to arrays and array manipulation
import matplotlib.pyplot as plt                                     #library with functions for visual plotting of data

import sys                                                          #generically useful library for low-level i/o
import queue                                                        #used in the callback function for InputStream
import threading                                                    #library for parralelism and multithreading; not strictly necessary for MFP but would be for anything higher fidelity, and something I'd like to learn better, so might as well

                                                                    #sounddevice would be replaced by the functionally similar pyaudio in a high-fidelity prototype capable of running on mobile devices through Kivy


is_recording = True                                                 #global bool that records current recording state; would be bound to a button in the UI in a higher fidelity prorotype

def threaded_recording(file_name,q):                                 #function for recording; intended to be run in parallel to allow for arbitrary recording duration

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
        with sf.SoundFile(file_name, mode='x', samplerate=44100, channels=1) as file:
            with sd.InputStream(samplerate=44100,channels=1,callback=callback):
                print('#'*80)
                print('Press enter to stop recording')
                print('#'*80)
                while is_recording:
                    file.write(q.get())
    except KeyboardInterrupt:
        print('\nRecording ended unexpectedly, recording may be incomplete: ' + file_name)
    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))

    return None


def record(filename):                                               #function for taking recordings; kind of complicated, will break down w/ further comments
    
    q = queue.Queue()                                               #queue object that temporarily stores audio data as it comes in

                                                                    #thread object that implements the parallel recording function
    recording = threading.Thread(target=threaded_recording, args=(filename,q,))

    recording.start()                                               #recording starts here

    input()                                                         #main execution blocked by user input

    global is_recording                                             #after user input recieved, the while loop in the thread is set to false, and it exits

    is_recording = False     
    
    return None
    
                                                                    #function that performs first processing steps
                                                                    #converts audio file to numpy array, filters out irrelevant frequencies
                                                                    #returns as numpy arrays: original signal, filtered signal, fft transform
                                                                    #returns sample rate as int
def preprocessing(file_name):                                       

    valid_types = ['wav','au','flac']

    file_title, file_type = file_name.rsplit(".",1)

    if(file_type not in valid_types):                               #checks if the provided file is a type soundfile can read, if not, first creates a copy in a format that it can, through pydub
        
        dub.AudioSegment.from_file(file_name).export(file_title + '.wav')
        file_name = file_title + '.wav'

    signal, sr = sf.read(file_name)                                 #reads audio file to np.array (signal), and returns sampling rate as an int (sr) 

    if(signal.ndim > 1):                                            #checks the number of channels in the audio file
        signal = np.add(signal[:,0],signal[:,1])                    #if the file is stereo, the array is converted from 2D to 1D

                                                                    #defines a 10th order band-pass filter between 20 and 1000Hz; excluding sounds outside the range of heart signals
    band_pass_filter = sig.butter(10,[20,1000],'bp',output='sos',fs=sr)

    filtered_signal = sig.sosfilt(band_pass_filter, signal)         #applies the filter to the signal

    filtered_signal = filtered_signal/np.max(filtered_signal)       #normalizes filtered signal to magnitude 1

    signal_fft = sp.fft.fft(filtered_signal)                        #returns fast fourier transform of the signal

    return signal, sr, filtered_signal, signal_fft


def display_signal(signal, sr, id, mode, peaks = None):             #generates a graphical display of the input signal, depending on the designated mode
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
        plt.axis([0,9,0,1000])                                      #restrics area plotted to 0,1000
        plt.title('Spectrogram: ' + id)
        plt.show()
        plt.clf()

    elif(mode == 'peaks'):

        t = np.linspace(0.0,signal.size/sr,signal.size)

        plt.plot(t,signal)
        plt.plot(peaks/sr,signal[peaks],'x')
        plt.grid()
        plt.ylabel('Amplitude')
        plt.xlabel('Time [s]')
        plt.title('Signal Peaks (' + id + ')')
        plt.show()
        plt.clf()

    return None

def find_peaks_r(signal, stop = 3, peaks = None):                   #finds peaks through recursive application of scipy.find_peaks
                                                                    #by default, recurs three layers
                                                                    #problem with this method is that, while it involves few/no arbitrary parameters
                                                                    #it's kind of a black box
                                                                    #predicting how this will function on a variety of signals is very much non-trivial
                                                                    #likely to be removed, as a consequence

    if(stop == 3 and peaks == None):
        peaks = signal

    if(0<stop):
        filtered_peaks, properties = sig.find_peaks(signal, height = max(np.abs(signal))/10)
        stop-=1
        return peaks[find_peaks_recur3(signal[filtered_peaks],stop,filtered_peaks)]
    else:
        return peaks

def find_peaks_hs(signal, sr):                                      #finds peaks through smoothed hilbert transform
        
    sig_env = np.abs(sig.hilbert(signal))                           #signal envelope, which is defined as the absolute value of the hilbert transform of a signal
    
    window = 'hanning'                                              #defines a window type and size for smoothing
    window_len = sr//15                                             #window size of 1/15th sample rate is largely arbitrary; from testing that seemed to produce fairly good smooth approximations of the signal envelope; returning singular peaks without distorting the signal's shape too much.
                                                                    #problem with this is that it's liable to do funky things with non-standard sampling rates; assuming the standard use case for the software is on signals it has recorded though, this is something easily controllable
                                                                    #(additionally, it's fairly arbitrary to resample as needed, if it comes up during testing)
    w=np.hanning(window_len)
                                                                    #generates an array that is a copy of the signal, padded to minimize boundary effects
    signal_s = np.r_[sig_env[window_len-1:0:-1],sig_env,sig_env[-2:-window_len-1:-1]]
    
    if(sr//15 % 2 == 0):                                            #checks to see how the sample rate divides, and tweaks the convolution output accordingly to make sure all the arrays stay the same length
        smooth_hilbert = np.convolve(w/w.sum(),signal_s,mode='valid')[(window_len//2-1):-(window_len//2)]
    else:
        smooth_hilbert = np.convolve(w/w.sum(),signal_s,mode='valid')[(window_len//2):-(window_len//2)]
    
                                                                    #uses scipy.find_peaks to find the peaks in the smoothed envelope
    env_peaks, properties = sig.find_peaks(smooth_hilbert,height=max(smooth_hilbert/10),width=[None,None])
    
    width_index = properties['width_heights']*sr//2                 #defines an array of peak widths that correspond to each envelope peak; on average the widths of the envelope peaks are a few samples longer than the true width of the signal's peaks, but the difference is slight
                                                                    #this corresponds roughly to the duration of the heart sound in question

    peak_value = np.empty((env_peaks.size-1))                       #some empty arrays that will become useful shortly
    peak_index = np.empty((env_peaks.size-1),dtype = int)           #discards first peak to avoid boundary errors

    i = 1

    while(i<peak_value.size+1):                                     #finds the value and position of the actual signal peak associated with the peak found in the smoothed anvelope
        peak_value[(i-1)] = np.max(signal[int(env_peaks[i]-width_index[i]):int(env_peaks[i]+width_index[i])])
        peak_index[(i-1)] = int(np.where(signal[int(env_peaks[i]-width_index[i]):int(env_peaks[i]+width_index[i])] == peak_value[(i-1)])[0] + env_peaks[i] - width_index[i])
        i+=1

    return peak_index, width_index, peak_value                                  #returns the location, duration and amplitude of peaks, for use in feature extraction

def feature_extraction(signal, sr):

    feature = {}                                                    #initializes empty dictionary, that will hold signal features as they are extracted                

                                                                    #calls find_peaks_hs, and stores the returns in the dictionary
    peak_index, width_index, peak_value = find_peaks_hs(signal,sr)

    feature.setdefault('Peak Indicies',peak_index)
    feature.setdefault('Peak Widths',width_index)
    feature.setdefault('Peak Amplitudes',peak_value)

    i = 0

    avg_diff = 0

    while(i<peak_index.size-1):
        avg_diff += (peak_index[i+1]-peak_index[i])
        i+=1

    avg_period = 2*avg_diff/(peak_index.size*sr)

    feature.setdefault('Heart Rate',(60/(avg_period)))              #stores the heart rate, defined as 60 divided by the average period (at 2 peaks/period, twice the average time between the peaks)

    #in progress: convert peak indicies to heartrate (DONE) - easy if I assume the majority of peaks are s1 or s2 (60 over twice the average distance between peaks); if anomalous peaks show up or the signal is too noisy though the number will be way off
    #sort peaks by amplitude; separate somehow? (DONE) - again, easy if I can assume the majority of peaks are either s1 or s2 (sort by amplitude, split in half?); more useful in the case where the signal is *not* so clean, however
    #breakdown of cardiac cycle, avg duration (DONE), amplitude (DONE), frequency content during s1, s2, systole, diastole

                                                                    #procedurally checks the relationship between adjacent peaks, assuming they will follow a repeating high/low pattern
                                                                    #gets around the issues with chuncking going wrong if heart rate was determined incorrectly, 
                                                                    #although it has some issues of it's own; namely that if A: multiple anomalies occur between expected peaks, it will go wrong, (this could likely be fixed without rewriting from scratch, although it will take some doing) 
                                                                    #or if some noise causes peak amplitudes to be inconsistent, it will flag things as anomalies that may not be

                                                                    #defines several useful variables; a counter, a placeholder for peak delta, a flag for the placement of the final index, and empty arrays that will contain the sorted indicies
    i = 0

    peak_delta = 0

    possible_anomalies = np.empty(0)

    s1 = np.empty(0)

    s2 = np.empty(0)

    last_encountered = 0


    while(i < peak_value.size-1):

        if(peak_delta == 0):

            peak_delta = peak_value[i+1] - peak_value[i]

            if(peak_delta > 0):
                s2 = np.append(s2,peak_index[i])
            else:
                s1 = np.append(s1,peak_index[i])

            i+=1

        elif(peak_value[i+1] - peak_value[i] > 0 and peak_delta < 0):

            peak_delta = peak_value[i+1] - peak_value[i]
            s2 = np.append(s2,peak_index[i])
            i+=1
            last_encountered = 2

        elif(peak_value[i+1] - peak_value[i] < 0 and peak_delta > 0):

            peak_delta = peak_value[i+1] - peak_value[i]
            s1 = np.append(s1,peak_index[i])
            i+=1
            last_encountered = 1

        else:

            peak_delta = peak_value[i+1] - peak_value[i]
            possible_anomalies = np.append(possible_anomalies,peak_index[i])
            i+=1

                                                                    #statement to handle the last indexed peak; could default either to anomaly, or assume correctness; assuming correctness for now
    if(last_encountered == 2):
        s1 = np.append(s1,peak_index[i])
    else:
        s2 = np.append(s2,peak_index[i])

                                                                    #stores sorted peaks into arrays in the dict
    feature.setdefault('S1 Peaks',s1)
    feature.setdefault('S2 Peaks',s2)
    feature.setdefault('Anomalous Peaks',possible_anomalies)

    avg_s1 = 0                                                      #defines placeholder for average s1 amplitude, and resets counter var to 0

    i = 0

    while(i < s1.size):
        avg_s1 += signal[int(s1[i])]
        i+=1

    avg_s1 /= s1.size

    avg_s2 = 0                                                      #above repeated for s2

    i = 0

    while(i < s2.size):
        avg_s2 += signal[int(s2[i])]
        i+=1

    avg_s2 /= s2.size

    feature.setdefault('Average S1/S2 Ratio', avg_s1/avg_s2)        #stores ratio between amplitudes of s1 and s2


    avg_s1 = 0                                                      #reset counter and placeholder to 0

    i = 0

    while(i<s1.size):                                               #summs the width in indexes of the sorted peaks
        avg_s1 += width_index[int(np.where(peak_value == signal[int(s1[i])])[0])+1]
        i+=1

    avg_s1 /= s1.size

    avg_s2 = 0                                                      #reset counter and placeholder to 0

    i = 0

    while(i<s2.size):
        avg_s2 += width_index[int(np.where(peak_value == signal[int(s2[i])])[0])+1]
        i+=1

    avg_s2 /= s2.size

    feature.setdefault('S1 Duration', avg_s1*2/sr)
    feature.setdefault('S2 Duration', avg_s2*2/sr)

    return feature

#record('testing.wav')                                              #recording demo  

file_name = 'Apex16(noisy).au'

processed_signal = preprocessing(file_name)                         #preprocessing demo
                                                                    #file is a heart sound (Apex16.au) overlaid with three sin waves, one at 10 hz, one at ~2000hz and one at ~1750hz

                                                                    #display demo
print('Displaying Input Phonocardiogram\n\n')                                                                    
display_signal(processed_signal[0],processed_signal[1],'Input','time')
print('Displaying Filtered Phonocardiogram\n\n')
display_signal(processed_signal[2],processed_signal[1],'Filtered','time')
print('Displaying Fourier Tranform of Filtered Signal\n\n')
display_signal(processed_signal[3],processed_signal[1],'FFT of Filtered Signal','freq')
print('Displaying Spectrogram of Filtered Signal\n\n')
display_signal(processed_signal[2],processed_signal[1],'Filtered Signal','spec')

                                                                    #feature extractiion demo

                                                                    #main function call
signal_features = feature_extraction(processed_signal[2],processed_signal[1])

print('Times of Peaks corresponding to S1 [s]')
print(signal_features['S1 Peaks']/processed_signal[1])              #reads out the indicies of peaks corresponding to s1 and s2
print('\n\nTimes of Peaks corresponding to S2 [s]')
print(signal_features['S2 Peaks']/processed_signal[1])
print('\n\nTimes of Anomalous Peaks [s] (Note: If not empty, other signal features may have been calculated incorrectly)')
print(signal_features['Anomalous Peaks']/processed_signal[1])

print('\n\nDisplaying Filtered Signal with Peaks Highlighted')      #displays signal with peaks highlighted
display_signal(processed_signal[2],processed_signal[1],'Filtered','peaks',signal_features['Peak Indicies'])

print('\n\nAverage Heart Rate (BPM)')
print(signal_features['Heart Rate'])                                #reads out determined heart rate
print('\n\nAverage Ratio Between Amplitudes of S1 and S2')
print(signal_features['Average S1/S2 Ratio'])                       #reads out ratio between amplitudes of S1 and S2
print('\n\nAverage Duration of S1 [s]')
print(signal_features['S1 Duration'])                               #reads out determined average durations of s1 and s2
print('\n\nAverage Duration of S2 [s]')
print(signal_features['S2 Duration'])

#file_title, file_type = file_name.rsplit(".",1)                    #uncomment if you want output of filtered audio file
#sf.write(file_title + 'filtered.wav',processed_signal[2],processed_signal[1])                 

#sd.play(processed_signal[2],processed_signal[1])                     #uncomment if you want to listen to the filtered audio
#sd.wait()




    







