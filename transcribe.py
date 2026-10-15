import json
import sys
import wave
import zipfile
from pathlib import Path

import requests
import soundfile as sf
import numpy as np
from scipy.signal import resample_poly
from vosk import Model, KaldiRecognizer, SetLogLevel

# ── Настройки ──────────────────────────────────────────────
INPUT_DIR = Path(__file__).parent / "voices"

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
MODEL_NAME = "vosk-model-small-ru-0.22"

MODEL_DIR = Path(__file__).parent / "model"
SAMPLE_RATE = 16000

SetLogLevel(-1)  # тишина от Vosk


# ── Модель ─────────────────────────────────────────────────
def ensure_model() -> str:
    """Скачивает и распаковывает модель в ./model/, если её нет."""
    model_path = MODEL_DIR / MODEL_NAME
    zip_path = MODEL_DIR / "model.zip"
    
    if model_path.exists():
        return str(model_path)

    MODEL_DIR.mkdir(exist_ok=True)

    if zip_path.exists():
        print(f"📂 Распаковываю существующий архив...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(MODEL_DIR)
        zip_path.unlink()
        print("✅ Модель готова!\n")
        return str(model_path)

    print(f"📦 Модель не найдена. Скачиваю (~40 MB)...")
    resp = requests.get(MODEL_URL, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(zip_path, "wb") as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                bar = downloaded * 30 // total
                pct = downloaded * 100 // total
                print(f"\r   [{'█' * bar}{'·' * (30 - bar)}] {pct}%", end="", flush=True)

    print("\n📂 Распаковываю...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(MODEL_DIR)
    zip_path.unlink()
    print("✅ Модель готова!\n")
    return str(model_path)


# ── Конвертация и очистка ──────────────────────────────────
def process_audio_for_vosk(ogg_path: str) -> str:
    """
    1. Качественно меняет частоту дискретизации (без артефактов np.interp).
    2. Применяет математический шумовой гейт для удаления фонового шума.
    3. Сохраняет в формат, идеальный для Vosk.
    """
    wav_path = str(Path(ogg_path).with_suffix(".wav"))
    
    # 1. Читаем аудио как float32 для точных математических операций
    audio, sr = sf.read(ogg_path, dtype='float32')
    
    # Если аудио стерео, делаем моно
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    
    # 2. Качественный ресемплинг (полифазный фильтр, стандарт индустрии)
    if sr != SAMPLE_RATE:
        audio = resample_poly(audio, SAMPLE_RATE, sr)
    
    # 3. Шумовой гейт на чистом NumPy (замена тихого шума на абсолютную тишину)
    # Порог -20 dB: всё, что тише этого уровня, считается фоновым шумом и обнуляется
    threshold_db = -20.0
    threshold_linear = 10 ** (threshold_db / 20.0)
    
    frame_length = int(SAMPLE_RATE * 0.02)  # 20 мс, как в профессиональных VAD
    hop_length = frame_length // 2
    
    output = np.copy(audio)
    for i in range(0, len(audio) - frame_length, hop_length):
        frame = audio[i:i + frame_length]
        rms = np.sqrt(np.mean(frame ** 2))  # Среднеквадратичная амплитуда (громкость)
        
        if rms < threshold_linear:
            output[i:i + frame_length] = 0.0  # Заменяем шум на тишину

    # 4. Конвертируем обратно в 16-bit PCM для Vosk
    audio_int16 = np.clip(output * 32767, -32768, 32767).astype('int16')
    sf.write(wav_path, audio_int16, SAMPLE_RATE, subtype='PCM_16')
    
    return wav_path


# ── Распознавание ──────────────────────────────────────────
def transcribe(audio_path: str) -> str:
    """Принимает .ogg или .wav, возвращает распознанный текст."""
    model_path = ensure_model()
    model = Model(model_path)
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    wav_path = None
    
    try:
        # Если это ogg, конвертируем и чистим шум
        if audio_path.lower().endswith(".ogg"):
            wav_path = process_audio_for_vosk(audio_path)
            process_path = wav_path
        else:
            process_path = audio_path

        wf = wave.open(process_path, "rb")
        chunks = []

        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if res.get("text"):
                    chunks.append(res["text"])

        final = json.loads(rec.FinalResult())
        if final.get("text"):
            chunks.append(final["text"])

        wf.close()
        return " ".join(chunks).strip()
    finally:
        # Удаляем временный WAV файл после обработки
        if wav_path and Path(wav_path).exists():
            Path(wav_path).unlink()


# ── Вспомогательные функции ────────────────────────────────
def get_target_files(limit: int = None) -> list[Path]:
    """Возвращает список .ogg файлов из папки voices, отсортированных по имени."""
    INPUT_DIR.mkdir(exist_ok=True)
    files = list(INPUT_DIR.glob("*.ogg"))
    
    if not files:
        return []
    
    files.sort(key=lambda p: p.name)
    
    if limit is not None:
        return files[-limit:]
    
    return files


def process_single_file(file_path: Path):
    """Обрабатывает один файл: транскрибирует и выводит результат в консоль."""
    print("-" * 60)
    print(f"📄 Файл: {file_path.name}")
    
    print("🎧 Распознаю...")
    text = transcribe(str(file_path))
    
    print(f"📝 Результат:\n{text if text else '🤷 Речь не распознана.'}\n")


# ── CLI (Командная строка) ─────────────────────────────────
if __name__ == "__main__":
    print("🔄 Проверка модели...")
    ensure_model()

    if len(sys.argv) == 1:
        files = get_target_files()
        if not files:
            print(f"📁 Папка '{INPUT_DIR}' пуста или не содержит .ogg файлов.")
            print("💡 Скопируйте голосовые сообщения в эту папку и запустите скрипт снова.")
            sys.exit(0)
        
        print(f"🎯 Найдено файлов: {len(files)}. Начинаю обработку по очереди...\n")
        for f in files:
            process_single_file(f)

    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        
        if arg.isdigit():
            limit = int(arg)
            files = get_target_files(limit=limit)
            if not files:
                print(f"📁 В папке '{INPUT_DIR}' нет файлов для обработки.")
                sys.exit(0)
            
            print(f"🎯 Обрабатываю последние {limit} файл(ов)...\n")
            for f in files:
                process_single_file(f)
        else:
            target_path = Path(arg)
            
            if not target_path.exists():
                alt_path = INPUT_DIR / arg
                if alt_path.exists():
                    target_path = alt_path
                else:
                    print(f"❌ Файл не найден: {arg}")
                    sys.exit(1)
            
            process_single_file(target_path)

    else:
        print("⚠️ Неверный формат команды.")
        print("\n📘 Использование:")
        print("  python transcribe.py          # обработать все .ogg в папке 'voices'")
        print("  python transcribe.py N        # обработать последние N файлов в папке 'voices'")
        print("  python transcribe.py file.ogg # обработать конкретный файл")
        sys.exit(1)