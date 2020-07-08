#does this just work?
#no
#suspect parser is the problem
#intends to be run from terminal; that's not the case here
#scrap parser, figure out something else
#queue and sys likely necessary
#tempfile likely not?
#time was used for testing, but isn't necessary for the actual project
#not strictly certain sys is necessary either, but I don't want to fiddle with callback functions too much; I know this works, would rather not hcange it unless I have to

import queue
import sys
import threading
import time

import sounddevice as sd
import soundfile as sf
import numpy as np  # Make sure NumPy is loaded before it is used in the callback
import scipy as sp
import matplotlib.pyplot as plt
import scipy.signal as sig

#import sockets
#import tqdm
#import os
#^modules for network transfer to a server + some fancy qol dodads; getting set up will be a pain; might have to rent a server or set up a virtual one on the 'pute... could be easier if I buy a pc, since I could just use that... hrm :P
#not going to worry about this atm though


#start small

#print(sd.query_devices()) # works, but confusing :P Gives me 14 different audio devices, just for my laptop
                           # however, devices are specifiable by string id, which will be unique to the mic Garret's bought, so that's fine

#let's see if the default one works

#myrecording = sd.rec(10*44100,samplerate=44100,channels=1)
#sd.wait()
#sd.play(myrecording,44100)
#sd.wait()

#sick
#shit works
#now
#can I write this using soundfile, or should I switch to pydub, since I know that works?

#sf.write('testfile.wav', myrecording, 44100)

#...it just works
#cool
#so, screw parsers, but otherwise, this should 100% work for recording

#lets quickly sketch out a proof of concept of arbitrary recording length in the morning, but it should be chill
#(I could just restrict recordings to 30 secs or something, but seems cheap)
#lets switch back to the main file, and see if we can't get arbitrary duration going.

#wait no

# first test; while(stdin.empty())

#no luck, guess we have to figure out async/await

#although I've learned a lot about asyncio, and it definitely seems useful; it seems to not be what I want here, unfortunately >>

#Might have to learn threading, joy

#def thread_func(text):
#    print('Thread ', text, ': starting')
#    time.sleep(2)
#    print('Thread ',text,': finishing')
#
#x = threading.Thread(target=thread_func, args=(1,), daemon=True)
#print('pre-thread')
#x.start()
#print('post-thread')

#Hokay. :P

#So. Threading works, that's low-level io out of the way. (Since the o part is pretty arbitrary, with obv exception to sockets and such, but we're not worrying about that yet, and possibly never will, we'll see. :P)

#next big hurdle is going to be figuring out processing/analysis with scipy

#let's start that, shall we?

sample, sr = sf.read('Apex16.au')

single_channel_sample = np.add(sample[:,0],sample[:,1])

print(np.size(single_channel_sample))

x = np.linspace(0.0,single_channel_sample.size/sr,single_channel_sample.size)

plt.plot(x,single_channel_sample)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.title('Input Signal')
plt.show()
plt.clf()

sample_fft = sp.fft.fft(single_channel_sample)

sample_power = np.abs(sample_fft)

sample_freq = sp.fft.fftfreq(single_channel_sample.size, d=(1/sr))

plt.plot(sample_freq[0:4350], 2.0/single_channel_sample.size * sample_power[0:4350])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.grid()
plt.title('Fourier Transform of Signal (0-500Hz)')
plt.show()
plt.clf()

#above first takes an audio files, and converts it from dual-channel to single channel
#(finalized version would check # of channels first, but I know this sample is dual-channel)
#then plots the amplitude/time variation of the audio array 
#(a phonocardiogram in essence; you can fairly easily make out the first and second heart sounds by examination)
#then it transforms the audio array from time domain to frequency, and plots power/frequency

#next step: filtering

#zero_500_fft = sample_fft.copy()
#zero_500_fft[np.abs(sample_freq) > 4350] = 0
#filtered_sample = sp.fft.ifft(zero_500_fft)
#
#plt.plot(x,filtered_sample)
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
#plt.show()
#plt.clf()

#above is a super rough manual filter; literally setting every frequency above ~500 to zero in the transform then inverting it back to a signal
#below is just a quick comparison of filtered vs original signal (moved below butterworth plot, now))
#sounds *okay,* there is some distortion though and it's worh noting that sd.play can only take real-valued arrays, and the return from ifft is complex



#try with butterworth filter?

sos = sig.butter(10,500,'lp',fs=sr,output='sos')
butter_filtered_sample = sig.sosfilt(sos,single_channel_sample)

plt.plot(x,butter_filtered_sample)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()
plt.title('10th Order Butterworth Low-Pass Filter at 500 Hz')
plt.show()
plt.clf()

#subjective coparison by sound:

#print('playing original signal')
#sd.play(single_channel_sample,sr)
#sd.wait()
#print('playing rough filtered signal')
#sd.play(np.real(filtered_sample),sr)
#sd.wait()
#print('playing butter filtered signal')
#sd.play(butter_filtered_sample,sr)
#sd.wait()

#tbh, not really all that much difference; both filtered samples are slightly muffled. 
#h/e, this may change once I introduce some noise
#first, something easy; constant frequency sin waves 

sin_880 = 0.3*np.sin(2*np.pi*880*x*sr) 
sin_1046 = 0.3*np.sin(2*np.pi*1046*x*sr) 

sin_combined = np.add(sin_880,sin_1046)

noisy_sample = np.add(sin_combined,single_channel_sample)

plt.plot(x,noisy_sample)
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Signal Resulting from Introduced Noise')
plt.show()

#sd.play(noisy_sample,sr)
#sd.wait()

#introduced hum renders heart sound near indistinguishable
#try filter?

filter_500hz = sig.sosfilt(sos,noisy_sample)

plt.plot(x,filter_500hz)
plt.grid()
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Denoised Signal')
plt.show()

sd.play(filter_500hz,sr)
sd.wait()

#this doesn't work
#shit
#this was supposed to be the easy one
#aight, let's see if we can't figure out why

