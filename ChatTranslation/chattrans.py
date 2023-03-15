import time
from io import BytesIO
import numpy as np
import soundfile as sf # 音声データのフォーマット変換
import pygame # 喋らせる
import speech_recognition as sr # speech認識
from googletrans import Translator # 翻訳
import whisper # 文字起こし
from gtts import gTTS # text to speech

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
    
# 音声から文字起こしする
result = model.transcribe(
    audio_fp32,
    verbose=True,
    language='japanese',
    beam_size=5,
    fp16=True,
    without_timestamps=True
    )

# 文字起こし結果を表示する
# print(result["text"])

# 英語に翻訳する
translator = Translator()
translated = translator.translate(result["text"], dest="en")
print(translated.text)

# テキストを音声に変換
mp3_fp = BytesIO()
tts = gTTS(translated.text, lang='en')
tts.write_to_fp(mp3_fp)
mp3_fp.seek(0)

# 喋らせる
pygame.init()
pygame.mixer.init()
pygame.mixer.music.load(mp3_fp, 'mp3')
pygame.mixer.music.play()
time.sleep(5)