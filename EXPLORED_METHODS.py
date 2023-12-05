import pyaudio
import pygame
import wave
import os

from gtts import gTTS
import speech_recognition as sr
from textblob import TextBlob

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy import displacy

#from summa import summarizer, keywords, textrank

#from gensim.summarization.summarizer import summarize

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.utils import get_stop_words



# initialize spacy 'en' model, keeping only tagger component needed for lemmatization
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


def record_audio():
    # Start recording the audio
    CHUNK = 1024 
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    ####################################################################
    RECORD_SECONDS = 15 #################
    
    WAVE_OUTPUT_FILENAME = "output.wav"

    audio = pyaudio.PyAudio()

    stream = audio.open(format=FORMAT, 
                        channels=CHANNELS,
                        rate=RATE, 
                        input=True,
                        frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()

    # Save the recorded audio
    f = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    f.setnchannels(CHANNELS)
    f.setsampwidth(audio.get_sample_size(FORMAT))
    f.setframerate(RATE)
    f.writeframes(b''.join(frames))
    f.close()

    return WAVE_OUTPUT_FILENAME




def transcribe_audio(audio_file):
    # Load the audio file
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_text = r.record(source)

    # Convert audio to text
    try:
        text = r.recognize_google(audio_text)
        return text
    except sr.UnknownValueError:
        return "Unable to recognize speech"
    except sr.RequestError as e:
        return "Error: {}".format(e)




def text_to_speech(text):
    # Convert text to speech
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")

    pygame.init()
    audio_file = 'output.mp3'
    pygame.mixer.music.load(audio_file)

    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.quit()



def summarize_text(text):
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summarizer.stop_words = get_stop_words("english")
    sentences = summarizer(parser.document, 3)
    summary = ""
    for sentence in sentences:
        summary += str(sentence)
    return summary



def extract_keywords(text):
    # Create a Doc object using spacy
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)

    # Extract the keywords
    keywords = []
    for token in doc:
        if not token.is_stop and token.is_alpha and token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            keywords.append(token.text)

    # Return the keywords
    return ", ".join(keywords)


    
    
def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    
    if sentiment_score < -0.5:
        sentiment_label = "Very Negative"
    elif sentiment_score >= -0.5 and sentiment_score < 0:
        sentiment_label = "Negative"
    elif sentiment_score == 0:
        sentiment_label = "Neutral"
    elif sentiment_score > 0 and sentiment_score < 0.5:
        sentiment_label = "Positive"
    else:
        sentiment_label = "Very Positive"
        
    return sentiment_score, sentiment_label



def get_parse(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    displacy.render(doc, style="dep", jupyter=True)



# Start the program

# Record the audio
print("Listening...")
audio_file = record_audio()
print("Finished recording...")


# Transcribe the audio
print("Transcribing audio...")
text = transcribe_audio(audio_file)
print("Transcribed text:", text)


# Perform sentiment analysis on the text
print("Analyzing sentiment...")
score, word = get_sentiment(text)
print("Sentiment:", score," ",word)


# Summarize the text
print("Summarizing text...")
summary = summarize_text(text)
print("Summary:", summary)


# Extract the keywords from the text
print("Extracting keywords...")
keywords = extract_keywords(text)
print("Keywords:", keywords)

    
#Visualize the sentence using displacy:
print("Displaying parse tree...")
get_parse(text)


# Convert the text to speech and play the audio
print("Converting text to speech...")
text_to_speech(text)

