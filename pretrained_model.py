import tensorflow as tf
import librosa
import numpy as np
import pyaudio
import time
from transformers import T5ForConditionalGeneration, T5Tokenizer

# pretrained txt to speech.
model = tf.keras.models.load_model('speech_model.h5')
speech_commands = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

#txt to speech and tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# audiostream mf 
audio = pyaudio.PyAudio()
stream = audio.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)

# speech to txt
def speech_to_text(prompt, timeout):
    print(prompt)
    start_time = time.time()
    while True:
        if time.time() - start_time > timeout:
            print('Timeout')
            return ''
        audio_data = stream.read(8000, exception_on_overflow=False)
        audio_signal = np.frombuffer(audio_data, dtype=np.int16)
        audio_signal = librosa.resample(audio_signal.astype(float), 16000, 8000, res_type='kaiser_fast')
        audio_signal = audio_signal.reshape(1, -1, 1)
        prediction = model.predict(audio_signal)
        command_index = np.argmax(prediction)
        if prediction[0][command_index] > 0.5:
            transcription = speech_commands[command_index]
            print(f'Transcription: {transcription}')
            return transcription

# txt to speec 
def text_to_speech(prompt):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = t5_model.generate(input_ids)
    speech = librosa.resample(output[0].numpy(), 22050, 16000, res_type='kaiser_fast')
    return speech

# input speech
while True:
    transcription = speech_to_text('Speak now:', 5)
    if transcription:
        speech = text_to_speech(transcription)
        stream.write(speech.astype(np.int16).tobytes())
