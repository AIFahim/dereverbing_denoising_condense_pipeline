from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import librosa
import librosa.display as display
import soundfile as sf
import os
import torch
import torchaudio
import cv2
from tqdm import tqdm
import glob
from pydub import AudioSegment
from argparse import ArgumentParser
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, random_split
import collections
import numpy as np
import contextlib
import sys
import glob
import wave
import csv
from tqdm import tqdm
from torch import chunk
import webrtcvad
import os
import librosa as lr
import shutil
import time
import json
import requests
from multiprocessing import Pool
import multiprocessing
from urllib3 import disable_warnings
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
from statistics import mean, variance
from scipy import stats
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dereverb.utils import generate_spec, reconstruct_wave, graph_spec
from dereverb.reverb_dataset import ReverbDataset
from dereverb.convolutional_models import UNetRev

errors = open("error.txt", "w+", encoding = 'UTF-8')

def eco_cancel(input_path:str ):
    print("Echo cancelling . . . ")
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"

    checkpoint_file = 'unet_state_dict.pth'
    model_unet = UNetRev(n_channels=1, bilinear=False, confine = False)
    model_unet.load_state_dict(torch.load(checkpoint_file))
    
    
    filename = (input_path.split('/')[-1]).split('.')[0]
    extension = '.'  + ((input_path.split('/')[-1]).split('.')[1])
    # print(filename)
    # print(extension)
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(input_path)

    n_fft = 2048
    win_length = None
    hop_length = 512

    # Define transform
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )

    spec = spectrogram(SPEECH_WAVEFORM)

    spec = spec.numpy()[0,...]
    spec = librosa.power_to_db(spec)
    
    rev_spec = spec

    rev_spec_expanded = np.expand_dims(rev_spec, axis=0)
    example = torch.tensor(rev_spec_expanded[None, :, :, :], dtype=torch.float32)

    model_unet.eval()

    with torch.no_grad():
        derev = model_unet(example)

    derev = derev.clone().detach().cpu().numpy()
    derev = derev[0, 0, :, :]


    griffin_lim = T.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )

    reconstructed_waveform = librosa.db_to_power(derev, ref=1.0)
    reconstructed_waveform = griffin_lim(torch.from_numpy(reconstructed_waveform))



    reconstructed_waveform = reconstructed_waveform.view(1, -1)

    # """ 
    # return reconstructed_waveform
    
    # out_path = f"/home/asif/Noise_Filtering_Works/Outputs/dereverb_files/{filename}_dereverb{extension}"

    out_path = f"/home/asif/Noise_Filtering_Works/dereverb_denoise_condenser/tts_test/{filename}_dereverb{extension}"
        
    torchaudio.save(out_path, reconstructed_waveform, sample_rate=SAMPLE_RATE)
    print("echo cancellation done. . . ")
    return out_path
    # """
    # return reconstructed_waveform

def deepfilternet(out_path:str):
    # os.environ["CUDA_VISIBLE_DEVICES"]=""
    # os.environ["CUDA_VISIBLE_DEVICES"]="1"

    print("denoising start . . . ")
    filename = out_path.split(os.sep)[-1].split('.')[0]
    # print("deep",filename)
    extension = '.' + (out_path.split(os.sep)[-1].split('.')[-1])
    # print(extension)
    
    
    # target_dir = '/home/asif/Noise_Filtering_Works/Outputs/denoised_files' 
    # target_dir = '/home/asif/Noise_Filtering_Works/Outputs/Final_denoised'
    target_dir ='/home/asif/Noise_Filtering_Works/dereverb_denoise_condenser/tts_test/Final_denoised'
    
    
    
    os.system(f'deepFilter "{out_path}" --output-dir {target_dir}')
    
    output_path = target_dir + '/' + filename  + '_DeepFilterNet2'  + extension
    # wav, sr = lr.load(path, sr = None)
    print("Done denoising")
    # count += 1
    
       
    return output_path

start_end_list = []
SILENCE_THRESHOLD_MILLISECONDS = 400

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
   
    pcm_data = AudioSegment.from_file(path).set_channels(1).set_sample_width(2).raw_data
    frame_rate = AudioSegment.from_file(path).frame_rate
    
    if frame_rate not in [8000, 16000, 32000, 48000]:
        # print("changing sample rate . . . . . to 16000")
        pcm_data = AudioSegment.from_file(path).set_channels(1).set_sample_width(2).set_frame_rate(16000).raw_data
        frame_rate = 16000
    return pcm_data, frame_rate 


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

    

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False
    start_end_list.clear()
    
    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        # sys.stdout.write('1' if is_speech else '0')

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                # sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp))
                start_end_list.append(ring_buffer[0][0].timestamp)
                # dic['chunk']
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                start_end_list.append(frame.timestamp + frame.duration)
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    
    
    if triggered:
        # sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        start_end_list.append(frame.timestamp + frame.duration)
    # sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])



def get_chunks(path):
    
    
    intermediate_silence_duration_ms_list = []

    filename = (path.split('/')[-1]).split('.')[0]
    extension = '.'  + ((path.split('/')[-1]).split('.')[1])  
    # print('filename', filename)
    # print('extension', extension)  
    audio, sample_rate = read_wave(path)
    vad = webrtcvad.Vad(1) # agressiveness
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)
    

    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    
    
    # dir = '/home/auishik/dereverb_denoise_condenser/saved_chunks'
    dir = '/home/asif/Noise_Filtering_Works/Outputs/saved_chunks'

    
    for i, segment in enumerate(segments):
        pass
       
    
    
    start_end_2d_list = [list(item) for item in list(np.reshape(start_end_list, (-1, 2)))]
    # print("start end 2d list. . . ", start_end_2d_list)
    final_list = [start_end_2d_list[0]]
    # print("final_list . . .", final_list)
    
    for i in range(0, len(start_end_2d_list) - 1):
        
        intermediate_silence_duration_ms = (start_end_2d_list[i+1][0] - start_end_2d_list[i][1]) * 1000
        # print('. . .. ', start_end_2d_list[i][0], final_list[-1][1])
        # print(intermediate_silence_duration_ms)
        intermediate_silence_duration_ms_list.append(intermediate_silence_duration_ms)
        intermediate_silence_duration_ms_list[4] = 70
        intermediate_silence_duration_ms_list[6] = 34
        # if intermediate_silence_duration_ms >= SILENCE_THRESHOLD_MILLISECONDS:
        #     # print('yes')
        #     final_list.append(start_end_2d_list[i])
        # else:
        #     # # print('no')
        #     final_list[-1][1] = start_end_2d_list[i][1]
    print("list int . .  .", intermediate_silence_duration_ms_list)
    average = mean(intermediate_silence_duration_ms_list) / len(intermediate_silence_duration_ms_list)
    print("avg . . .", average)
    
    variance_n = variance(intermediate_silence_duration_ms_list) / len(intermediate_silence_duration_ms_list)
    print("variance . . ", variance_n)
    
    zscore = stats.zscore(intermediate_silence_duration_ms_list)
    print("zscore .. ", zscore)
    
    # print("final_final_list", final_list) 
    
    output = AudioSegment.empty()
    original = AudioSegment.from_file(path)
    for segment in final_list:
        if(output.duration_seconds > 0.0):
            output = output.append(original[segment[0]*1000:segment[1]*1000], crossfade=0)
        else:
            output = output.append(original[segment[0]*1000:segment[1]*1000], crossfade=0)
    
    # target_dir = '/'.join(path.split('/')[:-1])
    target_dir = '/home/auishik/dereverb_denoise_condenser/condensed_files'
    # print("target dir", target_dir)
    # original_name = path.split('/')[-1].split('.')[0]
    output.export(f"{target_dir}/{filename}.flac", format="flac")
    # except:
    #     errors.write(path + '\n')
    
    start_end_list.clear() 
    start_end_2d_list.clear()
    final_list.clear()
       
       
def clean_data(input_path):
    path = eco_cancel(input_path)
    out_path = deepfilternet(path)
    # result = get_chunks(out_path)
    # return result
    
    
if __name__ == '__main__':
    # """
    for file in tqdm(glob.glob('/home/asif/Noise_Filtering_Works/dereverb_denoise_condenser/ttsinput/*')):
        exits_files = "/home/asif/Noise_Filtering_Works/dereverb_denoise_condenser/ttsout" # "/home/asif/Noise_Filtering_Works/Outputs/Final_denoised/"
        eco_cancel(file)
        """
        file_name = file.split(os.sep)[-1].split('.')[0]
        # print(file_name)
        # print(exits_files+file_name+"_DeepFilterNet2.mp3")
        if os.path.exists(exits_files+file_name+"_dereverb_DeepFilterNet2.mp3"):
            # clean_data(file)
            # print("yes")
            pass
        else:
            # print("No")
            try:
                eco_cancel(file)
                # clean_data(file)
            except:
                pass
        """
    # get_chunks('/home/auishik/dereverb_denoise_condenser/audio_files/silence.wav')

    # path = eco_cancel('/home/asif/Noise_Filtering_Works/Sound Files/otherformats/bleating.flac')
    # path = deepfilternet(path)
    

    
    # write_wave("/home/asif/Noise_Filtering_Works/Outputs", path, 16000)





    print('done')

    """

    exits_files = "/home/asif/Noise_Filtering_Works/dereverb_denoise_condenser/tts output test" # "/home/asif/Noise_Filtering_Works/Outputs/Final_denoised/"

    # file_name = file.split(os.sep)[-1].split('.')[0]
    # print(file_name)
    # print(exits_files+file_name+"_DeepFilterNet2.mp3")
    # if os.path.exists(exits_files+file_name+"_dereverb_DeepFilterNet2.mp3"):
    #     # clean_data(file)
    #     # print("yes")
    #     pass
    # else:
    #     # print("No")
    #     try:
    clean_data(file)
        # except:

    """