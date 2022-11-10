import os
from os.path import join, isdir, isfile, basename
import glob
import time
import json
import soundfile as sf
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from argparse import ArgumentParser
from tqdm import tqdm

from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN

tacotron2 = Tacotron2.from_hparams(source="/u/junkaiwu/speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts")
hifi_gan = HIFIGAN.from_hparams(source="/u/junkaiwu/speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder")

mel_output, mel_length, alignment = tacotron2.encode_text("Test Model")

waveforms = hifi_gan.decode_batch(mel_output)

sf.write("test.wav", waveforms.numpy().reshape(-1), 22050)

