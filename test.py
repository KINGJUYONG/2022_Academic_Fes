from pydub import AudioSegment

def audio_to_text(filename):
    import speech_recognition as sr
    r = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio_data = r.record(source)
    text = r.recognize_google(audio_data, language='ko-KR')
    return text

import pydub
import os
import subprocess
command = "ffmpeg -i {} -ab 160k -ac 2 -ar 44100 -vn -y {}".format("test.mp4",  "apple.wav")
subprocess.call(command, shell=True)
k = audio_to_text("apple.wav")
print(k)