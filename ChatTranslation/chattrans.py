from io import BytesIO
import numpy as np
import soundfile as sf
import speech_recognition as sr
from googletrans import Translator
import whisper
from gtts import gTTS
import pygame
import time
    

# Whisperモデルのロード
model = whisper.load_model("medium")

# 音声入力
r = sr.Recognizer()
with sr.Microphone(sample_rate=16_000) as source:
    print("なにか話してください")
    audio = r.listen(source)

    print("音声処理中 ...")
    wav_bytes = audio.get_wav_data()
    wav_stream = BytesIO(wav_bytes)
    audio_array, sampling_rate = sf.read(wav_stream)
    audio_fp32 = audio_array.astype(np.float32)
    
# 音声ファイルから文字起こしする
result = model.transcribe(
    # "./001-sibutomo.mp3",
    audio_fp32,
    verbose=True,
    language='japanese',
    beam_size=5,
    fp16=True,
    without_timestamps=True
    )

# 文字起こし結果を表示する
# print(result["text"])

# googletransで英語に翻訳する
translator = Translator()
translated = translator.translate(result["text"], dest="en")
print(translated.text)

# 喋らせる.VALL-Eに置き換えたい.
pygame.init()
pygame.mixer.init()
mp3_fp = BytesIO()
tts = gTTS(translated.text, lang='en')
tts.write_to_fp(mp3_fp)
mp3_fp.seek(0)
pygame.mixer.music.load(mp3_fp, 'mp3')
pygame.mixer.music.play()
time.sleep(5)