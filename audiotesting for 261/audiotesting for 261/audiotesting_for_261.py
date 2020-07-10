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
import pydub as dub

#import sockets
#import tqdm
#import os
#^modules for network transfer to a server + some fancy qol dodads; getting set up will be a pain; might have to rent a server or set up a virtual one on the 'pute... could be easier if I buy a pc, since I could just use that... hrm :P
#not going to worry about this atm though
#tomer is on this now

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

print(sample.ndim)

print(np.size(single_channel_sample))

x = np.linspace(0.0,single_channel_sample.size/sr,single_channel_sample.size)

#plt.plot(x,single_channel_sample)
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
#plt.grid()
#plt.title('Input Signal')
#plt.show()
#plt.clf()

sample_fft = sp.fft.fft(single_channel_sample)

sample_power = np.abs(sample_fft)

sample_freq = sp.fft.fftfreq(single_channel_sample.size, d=(1/sr))

#f, t, s = sig.spectrogram(single_channel_sample, sr)
#plt.pcolormesh(t, f, s, shading='gouraud')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [s]')
#plt.axis([0,9,0,1000])
#plt.show()
#plt.clf()

#plt.plot(sample_freq[0:sample.size*1000//(2*sr)], 2.0/single_channel_sample.size * sample_power[0:sample.size*1000//(2*sr)])
#plt.xlabel('Frequency [Hz]')
#plt.ylabel('Power')
#plt.grid()
#plt.title('Fourier Transform of Signal (0-500Hz)')
#plt.show()
#plt.clf()

#above first takes an audio files, and converts it from dual-channel to single channel
#(finalized version would check # of channels first, but I know this sample is dual-channel)
#then plots the amplitude/time variation of the audio array 
#(a phonocardiogram in essence; you can fairly easily make out the first and second heart sounds by examination)
#then it transforms the audio array from time domain to frequency, and plots power/frequency
#also testing the scipy.signal.spectrogram function here, seeing if that plots out something intelligible

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

sos500 = sig.butter(3,500,'lp',fs=sr,output='sos')
#butter_filtered_sample = sig.sosfilt(sos500,single_channel_sample)

#plt.plot(x,butter_filtered_sample)
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
#plt.grid()
#plt.title('10th Order Butterworth Low-Pass Filter at 500 Hz')
#plt.show()
#plt.clf()

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

sin_1760 = 0.3*np.sin(2*1760*np.pi*x) 
sin_2092 = 0.3*np.sin(2*np.pi*2092*x) 
sin_10 = 0.3*np.sin(2*10*np.pi*x)

sin_combined = np.add(sin_1760,sin_2092)

noisy_sample = np.add(sin_combined,single_channel_sample)
extra_noisy = np.add(noisy_sample,sin_10)
sin10_1760 = np.add(sin_10,sin_1760)
noisy_sig = np.add(sin10_1760,single_channel_sample)

sf.write('Apex16(noisy).wav', extra_noisy, sr)
file_change, new_rate = sf.read('Apex16(noisy).wav')

#element-wise addition resuslts in an overlay of the original signal with a deeply annoying hum
#seriously, don't bother listening unless you're debugging, it's unpleasant :P
#also did some testing here to see if data was lost converting to/from audio file to array, and it doesn't *seem to?*
#it was late at night when I tested this though, so hard to say if any irregularities were my goofs or actually problems

#plt.plot(x,extra_noisy)
#plt.grid()
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
#plt.title('Signal Resulting from Introduced Noise')
#plt.show()

#sd.play(noisy_sample,sr)
#sd.wait()

#introduced hum renders heart sound near indistinguishable
#try filter?
sos20_500 = sig.butter(10,[20,500],'bp',output='sos', fs=sr)

filter_500hz = sig.sosfilt(sos500,noisy_sample)
filter_20_500hz = sig.sosfilt(sos20_500, file_change)

#plt.plot(x,filter_500hz)
#plt.grid()
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
#plt.title('Denoised Signal (Low Pass)')
#plt.show()
#
#plt.plot(x,file_change)
#plt.grid()
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
#plt.title('Very Noisy Signal')
#plt.show()

#plt.plot(x,filter_20_500hz)
#plt.grid()
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
#plt.title('Denoised Signal (Band Pass)')
#plt.show()

#sd.play(filter_500hz,sr)
#sd.wait()

#plt.plot(x,sin_combined)
#plt.grid()
#plt.xlabel('Time (s)')
#plt.ylabel('Amplitude')
#plt.title('880Hz+1046Hz')
#plt.show()
#
#sd.play(single_channel_sample,sr)
#sd.wait()
#sd.play(noisy_sample,sr)
#sd.wait()
#sd.play(filter_500hz,sr)
#sd.wait()

#this doesn't work
#shit
#this was supposed to be the easy one
#aight, let's see if we can't figure out why

#cool, shit works now
#step 1 of processing, restrict sample to given band: complete

#let's transfer this over to mfp now

#step 2: feature extraction w/ find_peaks
# use distance between peaks to find period, from which can be derived hr, and can be used to identify stage of cardiac cycle for any given part of the signal
# possibly apply again to identify s2 and compare ratios, then maybe peak duration/frequencies?

filtered_peaks, junk = sig.find_peaks(filter_20_500hz, height = max(np.abs(filter_20_500hz))/10)
#filtered2_peaks, junk = sig.find_peaks(filter_20_500hz, height=(max(filter_20_500hz[filtered_peaks])/10), distance=(sr/10))
filtered2_5_peaks, junk = sig.find_peaks(filter_20_500hz[filtered_peaks])
repeated_peaks, junk = sig.find_peaks(filter_20_500hz[filtered_peaks[filtered2_5_peaks]])
#repeated_peaks2, junk = sig.find_peaks(filter_20_500hz[filtered_peaks[filtered2_5_peaks[repeated_peaks]]])
#repeated_peaks3, junk = sig.find_peaks(filter_20_500hz[filtered_peaks[filtered2_5_peaks[repeated_peaks[repeated_peaks2]]]])

#print(filtered2_peaks.size)
#print(filtered2_5_peaks.size)

#plt.plot(x,filter_20_500hz)
#plt.plot(filtered_peaks/sr, filter_20_500hz[filtered_peaks], 'x')
#plt.title('Peaks after first pass detection')
#plt.show()
#plt.clf()

#plt.plot(x,filter_20_500hz)
#plt.plot(filtered2_peaks/sr, filter_20_500hz[filtered2_peaks], 'x')
#plt.title('first pass modified with max height')
#plt.show()
#plt.clf()

#plt.plot(x,filter_20_500hz)
#plt.plot(filtered_peaks[filtered2_5_peaks]/sr, filter_20_500hz[filtered_peaks[filtered2_5_peaks]], 'x')
#plt.title('second pass; peaks of peaks')
#plt.show()
#plt.clf()
#
#plt.plot(x,filter_20_500hz)
#plt.plot(filtered_peaks[filtered2_5_peaks[repeated_peaks]]/sr, filter_20_500hz[filtered_peaks[filtered2_5_peaks[repeated_peaks]]], 'x')
#plt.title('Peak Detection Testing: 3 Layer recursive')
#plt.show()
#plt.clf()
#
#plt.plot(x,filter_20_500hz)
#plt.plot(filtered_peaks[filtered2_5_peaks[repeated_peaks[repeated_peaks2]]]/sr, filter_20_500hz[filtered_peaks[filtered2_5_peaks[repeated_peaks[repeated_peaks2]]]], 'x')
#plt.title('fourth pass;')
#plt.show()
#plt.clf()
#
#quick test

def find_peaks_recur3(signal,i,peaks):
    if(0<i):
        filtered_peaks, junk = sig.find_peaks(signal, height = max(np.abs(signal))/10)
        i-=1
        return peaks[find_peaks_recur3(signal[filtered_peaks],i,filtered_peaks)]
    else:
        return peaks
    
peak_values = find_peaks_recur3(filter_20_500hz,3,filter_20_500hz)

peak_indicies = np.empty(peak_values.size, dtype=int)

i = 0

while(i<peak_values.size):
    peak_indicies[i] = np.where(filter_20_500hz == peak_values[i])[0]
    i+=1

print(peak_indicies)

signal_period = (peak_indicies[1]-peak_indicies[0])/sr

print(signal_period)

heart_rate = 60/signal_period

print(heart_rate)

#this works, let's wee what the plot looks like

plt.plot(x,filter_20_500hz)
plt.plot(peak_indicies/sr, peak_values, 'x')
plt.title('Recursive Peak Detection')
plt.show()
plt.clf()

#so, this does roughly what I want, but I have a sneaking suspicion that it won't be valid for many signals, and getting s2 out is going to be a job in and of itself
#time to learn about hilbert transforms :P

hilb_sig = sig.hilbert(filter_20_500hz)

sig_env = np.abs(hilb_sig)

sos50 = sig.butter(10,20,'lp',output='sos', fs=sr)

smooth_bert = sig.sosfilt(sos50,sig_env)
#filtering like this gets me single peaks, but also introduces hella phase shift, and some minor attenuation

window = 'hanning'
windo_len = sr//10

signal_s = np.r_[sig_env[windo_len-1:0:-1],sig_env,sig_env[-2:-windo_len-1:-1]]

w=np.hanning(windo_len)

smoother_bert = np.convolve(w/w.sum(),signal_s,mode='valid')[(windo_len//2-1):-(windo_len//2)]
print(smoother_bert.size)

env_peaks, properties = sig.find_peaks(smoother_bert,height=max(smoother_bert/10),width=[None,None])

#print(properties)

#print(properties['width_heights'])

#print(properties['width_heights']*sr//2)

#print(env_peaks)

width_index = np.empty(properties['width_heights'].size, dtype=int)

width_index = properties['width_heights']*sr//2

#print(width_index)

plt.plot(x,filter_20_500hz)
#plt.plot(x,sig_env)
#plt.plot(x,smooth_bert)
plt.plot(x,2*smoother_bert)
plt.plot(env_peaks/sr,2*smoother_bert[env_peaks],'x')
plt.title('Signal Envelope and Peaks')
#plt.axis([1.25,1.75,-0.25,0.25])
plt.show()
plt.clf()

i = 0
avg_diff = 0

while(i<env_peaks.size-1):
    avg_diff += (env_peaks[i+1]-env_peaks[i])
    i+=1

avg_diff/=env_peaks.size*sr

#print(avg_diff)
#
#print(2*avg_diff)
#
#print(30/avg_diff)

#this...works? making the graphs pretty requires some tuning, but all that's really being lost is amplitude; other relevant info is pretty well conserved. 
#even w.r.t amplitude, the attenuation see should be *roughly* proportional? I think? so things like s1/s2 should be preserved
#the problem is coming up with a consistent definition for the window size. 1/10 of sample rate works for a relatively clean signal, but is basically arbitrary and doesn't work near as well for a noisy signal
#this kind of runs into the same problems as find-peaks; tuning is required and degree of feature preservation is dependent on the input signal
#mildly problematic
#i may just have to accept some level of distorion/feature loss though
#i may be able to get better data out by computing peak widths then finding the maxima over that section?


peak_amplitudes = np.empty(env_peaks.size, dtype=float)

i=0

while(i<peak_amplitudes.size):
    peak_amplitudes[i] = np.max(filter_20_500hz[int(env_peaks[i]-width_index[i]):int(env_peaks[i]+width_index[i])])
    i+=1

print(peak_amplitudes)

#this seems to work well; I like this

#so: heart rate done
# avg amp ratio is fairly trivial from here
#avg length systole vs avg legth diastole...
#easy for a clean signal, more complex for a noisy one
#frequency content of different stages should be doable (align peaks/troughs w/spectrogram )


#vic_cres_u, sr_vic = sf.read('Victoria-Cres-32.wav')
#
#vic_cres = sig.sosfilt(sos20_500,vic_cres_u)
#
#x2 = np.linspace(0.0,vic_cres.size/sr_vic,vic_cres.size)
#
#vic_bert = np.abs(sig.hilbert(vic_cres))
#
#windo_lenv = sr_vic//5
#
#vic_s = np.r_[vic_bert[windo_lenv-1:0:-1],vic_bert,vic_bert[-2:-windo_lenv-1:-1]]
#
#w=np.hanning(windo_lenv)
#
#smoother_vic = np.convolve(w/w.sum(),vic_s,mode='valid')[(windo_lenv//2-1):-(windo_lenv//2)]
#print(smoother_bert.size)
#
#vic_peaks, junk = sig.find_peaks(smoother_vic,height=max(smoother_vic/10))
#
#plt.plot(x2,vic_cres)
##plt.plot(x,sig_env)
##plt.plot(x,smooth_bert)
#plt.plot(x2,2*smoother_vic)
#plt.plot(vic_peaks/sr,2*smoother_vic[vic_peaks],'x')
#plt.title('Signal Envelope and Peaks (Signal from Physical Testing)')
##plt.axis([1.25,1.75,-0.25,0.25])
#plt.show()
#plt.clf()


#plt.plot(env_peaks/sr,sig_env[env_peaks],'x')
#plt.plot(x,sig_env)
#plt.title('Signal Envelope Peaks')
#plt.show()
#plt.clf()



#This... is less useful than hoped time to try something else

#first though, let's do some basic analysis of hte signal garret sent

#vic_m4a = dub.AudioSegment.from_file('Victoria-Cres-32.m4a')
#
#vic_m4a.export('Victoria-Cres-32.wav', format='wav')
#
#vic_cres, sr_vic = sf.read('Victoria-Cres-32.wav')

#... of course this is literally the first file format I've come accross to not work
# sick :P
# easy fix, at least

#x_new = np.linspace(0.0,vic_cres.size/sr,vic_cres.size)
#
#plt.plot(x_new,vic_cres)
#plt.xlabel('Time [s]')
#plt.ylabel('Amplitude')
#plt.title('Garret Sample (Unfiltered)')
#plt.grid()
#plt.show()
#plt.clf()
#
#vic_filt = sig.sosfilt(sos20_500,vic_cres)
#
#sf.write('Victoria-Cres-32(Filtered).wav',vic_filt,sr_vic,format='wav')
#
#temp = dub.AudioSegment.from_file('Victoria-Cres-32(Filtered).wav')
#temp.export('Victoria-Cres-32(Filtered).mp3',format='mp3')
#
#plt.plot(x_new,vic_filt)
#plt.xlabel('Time [s]')
#plt.ylabel('Amplitude')
#plt.title('Garret Sample (Filtered)')
#plt.grid()
#plt.show()
#plt.clf()

#sd.play(vic_filt,sr_vic)
#sd.wait()

#vic_pval = find_peaks_recur3(vic_filt,3,vic_filt)
#
#vic_ppos = np.empty(vic_pval.size, dtype=int)
#
#i = 0
#
#while(i<vic_pval.size):
#    vic_ppos[i] = np.where(vic_filt == vic_pval[i])[0]
#    i+=1
#
#plt.plot(x_new,vic_filt)
#plt.plot(vic_ppos/sr_vic, vic_pval, 'x')
#plt.title('Recursive Peak Detection (Garret Sample)')
#plt.show()
#plt.clf()
#
#f, t, s = sig.spectrogram(vic_cres, sr_vic)
#plt.pcolormesh(t, f, s, shading='gouraud')
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [s]')
#plt.axis([0,12,0,1000])
#plt.show()
#plt.clf()

#plt.plot(x,filter_20_500hz)
#plt.plot(filtered_peaks[filtered2_5_peaks[repeated_peaks[repeated_peaks2[repeated_peaks3]]]]/sr, filter_20_500hz[filtered_peaks[filtered2_5_peaks[repeated_peaks[repeated_peaks2[repeated_peaks3]]]]], 'x')
#plt.title('fifth pass; peaks of peaks of peaks of peaks of peaks')
#plt.show()
#plt.clf()

#at layer 5 feature loss occurs (s2 no longer detected), so I will need to set some domain restrictions to get the signal I want
#if I restrict domain starting at layer 1; feature loss occurs at pass 3, however pass 3 nicely gives me the positions of s1, which is interesting
#this will allow me to find period
#which should allow me to apply peak detection over restricted domains to find s2?

#so, okay
#dunno if my logic holds up/can be extrapolated to all signals, but basically I'm assuming that: any peak of amplitude less than a tenth that of the highest can be dismissed as background noise
#each time peak detection is recursively run, it restricts the domains of possible peaks; first finding each point where the adjacent points are less, then each point where each adjacent peak is less, and so on and so forth
#need to test if this method works for a variety of samples
#if I wanted to do this properly I should probably have computed the signal envelope and used that for peak detection, but I don't honestly know how to do that. >>
#it's something I'll research and attempt to implement if I have time.
#for now: