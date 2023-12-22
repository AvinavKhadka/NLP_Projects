import tensorflow as tf
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from pathlib import Path

#
# Set up the model and tokenizer
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")


#path and shit
input_file_path = Path("speechmothafucka.mp3")
output_file_path = Path("transcription.wav")


# 16kh wav file
audio = AudioSegment.from_file(input_file_path, format=input_file_path.suffix[1:])
audio.export("temp.wav", format="wav")
y, sr = librosa.load("temp.wav", sr=16000)


# split audio 10 sec transcription
chunk_size = 10 * sr
n_chunks = int(np.ceil(len(y) / chunk_size))
transcriptions = []
for i in range(n_chunks):
    start = i * chunk_size
    end = start + chunk_size
    chunk = y[start:end]
    
    # Spech to txt
    ###MAIN THING HAPPENS HERE BASICALLY
    ######
    ####
    input_values = tokenizer(chunk, return_tensors="tf").input_values
    logits = model(input_values).logits
    predicted_ids = tf.argmax(logits, axis=-1).numpy()
    transcription = tokenizer.decode(predicted_ids[0])
    
    transcriptions.append(transcription)
    ####
    ####


# Combine in string
transcription = "\n".join(transcriptions)


# txt file
with open("transcription.txt", "w") as f:
    f.write(transcription)


# back to audio and save
inputs = tokenizer(transcription, return_tensors="tf").input_ids
logits = model(inputs).logits
audio = logits.numpy()[0]
audio = librosa.resample(audio, 16000, sr)
sf.write(output_file_path, audio, 16000)
