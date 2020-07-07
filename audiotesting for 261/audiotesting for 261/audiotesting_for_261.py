#does this just work?
#no
#suspect parser is the problem
#intends to be run from terminal; that's not the case here
#scrap parser, figure out something else
#queue and sys likely necessary
#tempfile likely not?

import queue
import sys
import asyncio

import sounddevice as sd
import soundfile as sf
import numpy  # Make sure NumPy is loaded before it is used in the callback
assert numpy  # avoid "imported but unused" message (W0611)

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



async def record_until(loop, num, is_recording):
    if(is_recording):
        num+=1
    else: loop.stop()

def main():
    
    input('Press enter to stop recording.')
    is_recording = False
    loop.close()


#next big hurdle is going to be figuring out processing/analysis with scipy

#q = queue.Queue()


#def callback(indata, frames, time, status):
#    """This is called (from a separate thread) for each audio block."""
#    if status:
#        print(status, file=sys.stderr)
#    q.put(indata.copy())


#try:
#    if args.samplerate is None:
#        device_info = sd.query_devices(args.device, 'input')
#        # soundfile expects an int, sounddevice provides a float:
#        args.samplerate = int(device_info['default_samplerate'])
#    if args.filename is None:
#        args.filename = tempfile.mktemp(prefix='delme_rec_unlimited_',
#                                        suffix='.wav', dir='')
#
#    # Make sure the file is opened before recording anything:
#    with sf.SoundFile(args.filename, mode='x', samplerate=args.samplerate,
#                      channels=args.channels, subtype=args.subtype) as file:
#        with sd.InputStream(samplerate=args.samplerate, device=args.device,
#                            channels=args.channels, callback=callback):
#            print('#' * 80)
#            print('press Ctrl+C to stop the recording')
#            print('#' * 80)
#            while True:
#                file.write(q.get())
#except KeyboardInterrupt:
#    print('\nRecording finished: ' + repr(args.filename))
#    parser.exit(0)
#except Exception as e:
#    parser.exit(type(e).__name__ + ': ' + str(e))


