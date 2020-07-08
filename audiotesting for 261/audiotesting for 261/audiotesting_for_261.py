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
plt.show()
plt.clf()

sample_fft = sp.fft.fft(single_channel_sample)

sample_power = np.abs(sample_fft)

sample_freq = sp.fft.fftfreq(single_channel_sample.size, d=(1/sr))

sample_x = np.linspace(0.0,1000,1000)

#print(np.size(sample_fft,0))
#print(np.size(sample_power,0))
#print(np.size(sample_freq,0))
#print(sample_power)

plt.figure(figsize=(6,5))
plt.plot(sample_freq[0:2000], 2.0/single_channel_sample.size * sample_power[0:2000])
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power')
plt.grid()
plt.show()
plt.clf()

#num = 600
#t = 1.0/800.0
#x = np.linspace(0.0, num*t, num)
#y = np.sin(50.0*2.0*np.pi*x) + 0.5*np.sin(80.0*2.0*np.pi*x)
#y_fft = sp.fft.fft(y)
#x_fft = np.linspace(0.0,1.0/(2.0*t),num//2)
#plt.plot(x_fft,2.0/num * np.abs(y_fft[0:num//2]))
#plt.grid()
#plt.show()