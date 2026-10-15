"""
Bamanankan TTS (محسّن ومثبت) — Streamlit app
تحسينات إضافية لثبات التشغيل ودعم نصوص طويلة، أصوات متعددة، استنساخ صوت أفضل، ومعالجة أخطاء تحميل النموذج.
- دعم تحديد model_id عبر متغير بيئة أو واجهة المستخدم.
- إعادة محاولات (retries) عند فشل تحميل النموذج.
- كتابة ملف WAV آمنة، تحويل stereo->mono، والتحقق قبل الدمج.
- زر Demo لاختبار سريع، رسائل خطأ وإرشاد واضح.
- حماية واسعة (try/except) حول مواضع الفشل المحتملة.
"""

import os
import re
import io
import time
import tempfile
import torch
import numpy as np
import soundfile as sf
import streamlit as st
from typing import List, Tuple, Optional

# Optional: resemblyzer extraction
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    res_encoder = VoiceEncoder()
except Exception:
    res_encoder = None

# Transformers VITS interfaces
from transformers import VitsTokenizer, VitsModel

# ==========================================
# Configuration & utilities
# ==========================================
DEFAULT_MODEL = os.environ.get("MODEL_ID", "sudoping01/bambara-tts")

# Basic replacements and normalizer (unchanged core logic)
BASE_REPLACEMENTS = {
    r'\b100\b': 'kɛmɛ',
    r'\bG7\b': 'Gwolonwula',
    r'\b7\b': 'wolonwula',
    r'\b2027\b': 'bafila ani mugan ni wolonwula',
    r'\b2026\b': 'bafila ani mugan ni wɔɔrɔ',
    r'\b2029\b': 'bafila ani mugan ni kɔnɔntɔn',
    r'\b2030\b': 'bafila ani bisaba',
    r'\b4nan\b': 'naaninan',
    r'\b1000000000000\b': 'tiriliyɔni kelen',
    r'\b1000000000\b': 'miliyari kelen',
    r'\b1000000\b': 'miliyɔn kelen',
    r'\b1000\b': 'bakelen',
    r'\b100\b': 'kɛmɛ',
    r'\b10\b': 'tan',
    r'\b0\b': 'fu',
    r'\bcogo\b': 'cogo',
    r'\bcogoya\b': 'cogoya',
}

# ==========================================
# Normalizer functions
# ==========================================

def pronounce_number_decimal_percent(text: str) -> str:
    text = re.sub(r'(\d+)[,\.](\d+)%', r"kɛmɛsarada la \1 n'a kunkanfɛn \2", text)
    text = re.sub(r'(\d+)%', r'kɛmɛsarada la \1', text)
    def dec_rep(m):
        intp, fracp = m.group(1), m.group(2)
        return f"{intp} n'a kunkanfɛn {fracp}"
    text = re.sub(r'(\d+)[,\.](\d+)', dec_rep, text)
    return text


def pronounce_time(text: str) -> str:
    text = re.sub(r'(\d{1,2})h(\d{2}):(\d{2})',
                  lambda m: f"nɛgɛ kanɲɛ {int(m.group(1))} tɛmɛnen ye ni sanga {int(m.group(2))} ye ani segɔni {int(m.group(3))}",
                  text)
    text = re.sub(r'(\d{1,2}):(\d{2}):(\d{2})',
                  lambda m: f"nɛgɛ kanɲɛ {int(m.group(1))} tɛmɛnen ye ni sanga {int(m.group(2))} ye ani segɔni {int(m.group(3))}",
                  text)
    text = re.sub(r'(\d{1,2})h\b', lambda m: f"nɛgɛ kanɲɛ {int(m.group(1))}", text)
    return text

G2P_MAP = {"ɛ": "ɛ", "ɔ": "ɔ"}

def g2p_phonemize(text: str) -> str:
    for k, v in G2P_MAP.items():
        text = text.replace(k, v)
    return re.sub(r'\s+', ' ', text).strip()


def convert_bambara_numbers_and_text(text: str) -> str:
    text = text.strip()
    text = pronounce_time(text)
    text = pronounce_number_decimal_percent(text)
    for pattern, repl in BASE_REPLACEMENTS.items():
        text = re.sub(pattern, repl, text)
    def year_rep(m):
        year = int(m.group(0))
        mapping = {2026: 'bafila ani mugan ni wɔɔrɔ', 2027: 'bafila ani mugan ni wolonwula', 2029: 'bafila ani mugan ni kɔnɔntɔn', 2030: 'bafila ani bisaba'}
        return mapping.get(year, f"bafila ani mugan ni {year}")
    text = re.sub(r'\b20\d{2}\b', year_rep, text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = g2p_phonemize(text)
    return text

# ==========================================
# Chunking for long texts
# ==========================================

def chunk_text_smart(text: str, max_chars: int = 400, overlap: int = 60) -> List[str]:
    sentences = re.split(r'(?<=[.!?؟])\s+', text)
    chunks: List[str] = []
    cur = ""
    for s in sentences:
        if not cur:
            cur = s
        elif len(cur) + 1 + len(s) <= max_chars:
            cur = cur + " " + s
        else:
            chunks.append(cur.strip())
            overlap_words = max(1, overlap // 5)
            words = cur.split()
            cur = (" ".join(words[-overlap_words:]) + " " + s) if overlap_words > 0 else s
    if cur:
        chunks.append(cur.strip())
    # Ensure no empty chunks
    return [c for c in chunks if c]

# ==========================================
# Safe audio utilities
# ==========================================

def to_mono(wave: np.ndarray) -> np.ndarray:
    if wave is None:
        return np.array([], dtype=np.float32)
    if isinstance(wave, np.ndarray) and wave.ndim == 2:
        return wave.mean(axis=1).astype(np.float32)
    return np.asarray(wave).astype(np.float32)


def safe_concat(segments: List[np.ndarray]) -> np.ndarray:
    valid = [to_mono(s) for s in segments if s is not None and len(s) > 0]
    if not valid:
        return np.array([], dtype=np.float32)
    try:
        return np.concatenate(valid)
    except Exception:
        # fallback: pad/trim to smallest length then concatenate
        min_len = min(len(v) for v in valid)
        trimmed = [v[:min_len] for v in valid]
        return np.concatenate(trimmed)


def safe_write_wav(path: str, waveform: np.ndarray, sr: int):
    tmpfd, tmppath = tempfile.mkstemp(suffix=".wav")
    os.close(tmpfd)
    try:
        sf.write(tmppath, waveform, sr)
        os.replace(tmppath, path)
    finally:
        if os.path.exists(tmppath):
            try:
                os.remove(tmppath)
            except Exception:
                pass

# ==========================================
# Load model with retries and configurable model_id
# ==========================================
@st.cache_resource
def load_bambara_tts(model_id: Optional[str] = None, max_retries: int = 2, retry_delay: float = 2.0):
    model_id = model_id or os.environ.get("MODEL_ID", DEFAULT_MODEL)
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            tokenizer = VitsTokenizer.from_pretrained(model_id)
            model = VitsModel.from_pretrained(model_id)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            return tokenizer, model, device
        except Exception as e:
            last_exc = e
            time.sleep(retry_delay)
    raise RuntimeError(f"Failed to load model '{model_id}': {last_exc}")

# ==========================================
# Speed change util (simple resampling via interpolation)
# ==========================================

def change_speed(waveform: np.ndarray, speed: float) -> np.ndarray:
    if speed == 1.0 or waveform.size == 0:
        return waveform
    orig_len = waveform.shape[0]
    new_len = max(1, int(orig_len / speed))
    x_old = np.linspace(0, 1, orig_len)
    x_new = np.linspace(0, 1, new_len)
    new_wave = np.interp(x_new, x_old, waveform).astype(np.float32)
    return new_wave

# ==========================================
# Synthesis wrapper with guards
# ==========================================

def synthesize_chunks(tokenizer, model, device, chunks: List[str], speaker_embedding: Optional[np.ndarray] = None, style_speed: float = 1.0) -> Tuple[np.ndarray, int]:
    sampling_rate = getattr(model.config, "sampling_rate", 22050)
    audio_segments: List[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        try:
            inputs = tokenizer(chunk, return_tensors="pt").to(device)
            with torch.no_grad():
                # try model-specific generate_speech interface if available
                try:
                    if speaker_embedding is not None and hasattr(model, "generate_speech"):
                        output = model.generate_speech(**inputs, speaker_embedding=torch.tensor(speaker_embedding).to(device))
                    else:
                        out = model(**inputs)
                        output = getattr(out, "waveform", out)
                except Exception:
                    out = model(**inputs)
                    output = getattr(out, "waveform", out)

                if isinstance(output, torch.Tensor):
                    waveform = output.squeeze().cpu().numpy()
                else:
                    waveform = np.asarray(output)
        except Exception as e:
            st.warning(f"خطأ أثناء توليد الشريحة {i+1}: {e}")
            waveform = np.zeros(int(sampling_rate * 0.2), dtype=np.float32)

        waveform = to_mono(waveform)
        if speaker_embedding is None and style_speed != 1.0:
            try:
                waveform = change_speed(waveform, style_speed)
            except Exception:
                pass
        if waveform.size == 0:
            waveform = np.zeros(int(sampling_rate * 0.2), dtype=np.float32)
        audio_segments.append(waveform)
        audio_segments.append(np.zeros(int(sampling_rate * 0.18), dtype=np.float32))

    final_audio = safe_concat(audio_segments)
    return final_audio, sampling_rate

# ==========================================
# Streamlit UI
# ==========================================
st.set_page_config(page_title="Bamanankan TTS & Stable", layout="wide")
st.title("Bamanankan TTS — نسخة مستقرة")

col_top = st.columns([3,1])
with col_top[0]:
    model_id_input = st.text_input("Model ID (Hugging Face) - يمكنك تركه فارغاً لاستخدام الافتراضي:", value=os.environ.get("MODEL_ID", DEFAULT_MODEL))
with col_top[1]:
    if res_encoder is None:
        st.caption("resemblyzer غير مثبت — الاستنساخ الدقيق لن يعمل بدونها")

text_input = st.text_area("أدخل النص أو الأرقام بالبامبارا:", height=280, value="Aw ni ce. Nin ye bamanankan. I ka kɛnɛ wa? 2027 ani 18h02:01.")

col1, col2 = st.columns([2,1])
with col1:
    speaker_choice = st.selectbox("اختر نبرة/أسلوب الصوت:",
                                  ["Bourama (male, steady)", "Adama (male, natural)", "Moussa (male, clear)", "Amadou (male, warm)", "Intellectual (male, slow)", "Poet (female, soft)", "Female (general)", "Child (youthful)"])
with col2:
    ref_audio = st.file_uploader("رفع عيّنة صوتية للاستنساخ (اختياري, WAV/MP3)", type=["wav", "mp3"])

advanced = st.expander("إعدادات متقدمة")
with advanced:
    max_chars = st.number_input("أقصى طول قطعة (حرف)", min_value=100, max_value=5000, value=800, step=50)
    overlap = st.number_input("تراكب عند التجزئة (حرف تقريبي)", min_value=0, max_value=500, value=80, step=10)
    speed_map_note = st.checkbox("تمكين محاكاة الأسلوب عبر تغيير السرعة/النبرة (تقريبي)", value=True)

VOICE_STYLE_SPEED = {"Bourama (male, steady)": 0.98, "Adama (male, natural)": 1.0, "Moussa (male, clear)": 1.02, "Amadou (male, warm)": 0.97, "Intellectual (male, slow)": 0.92, "Poet (female, soft)": 1.03, "Female (general)": 1.06, "Child (youthful)": 1.12}

# Load model button (allows reloading with selected model_id)
if st.button("تحميل النموذج / إعادة تحميل"):
    try:
        with st.spinner("تحميل النموذج..."):
            tokenizer, model, device = load_bambara_tts(model_id=model_id_input)
        st.success("تم تحميل النموذج بنجاح")
    except Exception as e:
        st.error(f"فشل تحميل النموذج: {e}")
        tokenizer = model = device = None

# Quick demo
if st.button("تشغيل تجربة سريعة (Demo)"):
    try:
        tokenizer, model, device = load_bambara_tts(model_id=model_id_input)
        demo_text = "Aw ni ce. I ka kɛnɛ wa?"
        norm = convert_bambara_numbers_and_text(demo_text)
        audio, sr = synthesize_chunks(tokenizer, model, device, [norm])
        safe_write_wav("demo_output.wav", audio, sr)
        st.audio("demo_output.wav")
    except Exception as e:
        st.error(f"فشل demo: {e}")

# Generate main
if st.button("توليد صوت" ):
    if not text_input.strip():
        st.warning("الرجاء إدخال نص أولاً!")
    else:
        try:
            tokenizer, model, device = load_bambara_tts(model_id=model_id_input)
        except Exception as e:
            st.error(f"لا يمكن تحميل النموذج: {e}")
            tokenizer = model = device = None
        if tokenizer is None or model is None:
            st.stop()

        with st.spinner("جاري المعالجة..."):
            normalized_text = convert_bambara_numbers_and_text(text_input)
            chunks = chunk_text_smart(normalized_text, max_chars=int(max_chars), overlap=int(overlap))

            speaker_emb = None
            if ref_audio is not None:
                if res_encoder is not None:
                    try:
                        wav_bytes = ref_audio.read()
                        buf = io.BytesIO(wav_bytes)
                        wav = preprocess_wav(buf)
                        speaker_emb = res_encoder.embed_utterance(wav)
                    except Exception as e:
                        st.warning(f"فشل استخراج تمثيل المتحدث: {e}")
                        speaker_emb = None
                else:
                    st.info("resemblyzer غير مثبت — تحميل العيّنة لن يُستخدم للاستنساخ")

            style_speed = 1.0
            if speed_map_note and speaker_emb is None:
                style_speed = VOICE_STYLE_SPEED.get(speaker_choice, 1.0)

            progress = st.progress(0)
            final_segments: List[np.ndarray] = []
            sr = getattr(model.config, "sampling_rate", 22050)
            for idx, chunk in enumerate(chunks):
                chunk_audio, _ = synthesize_chunks(tokenizer, model, device, [chunk], speaker_embedding=speaker_emb, style_speed=style_speed)
                final_segments.append(chunk_audio)
                progress.progress(int((idx+1)/len(chunks)*100))

            final_audio = safe_concat(final_segments)
            if final_audio.size == 0:
                st.error("لم يتم توليد أي صوت. تأكد من إعداد النموذج أو جرّب نصاً أقصر.")
            else:
                out_file = "bambara_stable_output.wav"
                safe_write_wav(out_file, final_audio, sr)
                st.success("تم التوليد — استمع أو نزّل الملف:")
                st.audio(out_file)
                with open(out_file, "rb") as f:
                    st.download_button("تحميل WAV", f, file_name=out_file, mime="audio/wav")

                st.info("لتحسين الاستنساخ: ارفع عيّنة صوتية حقيقية وواضحة (عدة جمل) وتأكد من تثبيت resemblyzer).")

# Footer/help
st.markdown("---")
st.caption("نصائح: استخدم نموذج متعدد المتكلمين إن كنت تريد استنساخ صوت دقيق. ضع MODEL_ID في متغير البيئة أو في حقل Model ID.")
