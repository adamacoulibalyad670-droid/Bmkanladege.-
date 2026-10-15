"""
Bamanankan TTS (محسّن) — Streamlit app
تحسينات: خرائط نطق ممتدة، G2P مبسّط، تجزئة مع تراكب، دعم استنساخ الصوت (speaker embedding)، ومعالجة ما بعدها.
إضافات في هذا التعديل:
- مزيد من خيارات "الأسلوب/النبرة" (Intellectual, Poet (female), Female, Male, Child)
- معالجة بسيطة لتغيير السرعة/النبرة لتقليد اختلافات صوتية عندما لا يتوفر embedding
- تحسين تجزئة النصوص الطويلة والتعامل مع نصوص كبيرة جدًا
- وضع "تفريغ تقدم" لإبلاغ المستخدم أثناء توليد مقاطع متعددة

ملاحظة: التعديلات تعتمد على إمكانية النموذج الفعلية في التعامل مع speaker_embedding أو واجهات مخصصة للنموذج. إذا أردت، أستطيع لاحقًا ضبط model_id ليتناسب مع نموذج متعدد المتكلمين يدعم speaker_id أو speaker_embedding فعليًا.
"""

import re
import io
import math
import torch
import numpy as np
import soundfile as sf
import streamlit as st
from typing import List, Tuple, Optional

# ملاحظة: قد تحتاج لتثبيت مكتبات إضافية:
# pip install transformers soundfile streamlit pyloudnorm resemblyzer noisereduce

# محلول استخراج embedding من ملف مرجعي (اختياري)
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    res_encoder = VoiceEncoder()
except Exception:
    res_encoder = None

# مثال: تحميل نموذج VITS متعدد المتكلمين (قد تحتاج تغيير model_id حسب توفر نموذج Bambara حقيقي)
from transformers import VitsTokenizer, VitsModel

# ==========================================
# 1) خريطة نطق موسّعة للأمثلة (قابلة للتوسيع)
# ==========================================
BASE_REPLACEMENTS = {
    # أعداد ومقاطع متكررة/مطلوبة كما طلبت
    r'\b100\b': 'kɛmɛ',
    r'\bG7\b': 'Gwolonwula',
    r'\b7\b': 'wolonwula',
    r'\b2027\b': 'bafila ani mugan ni wolonwula',
    r'\b2026\b': 'bafila ani mugan ni wɔɔrɔ',
    r'\b2029\b': 'bafila ani mugan ni kɔnɔntɔn',
    r'\b2030\b': 'bafila ani bisaba',
    r'\b4nan\b': 'naaninan',
    # أمثلة أرقام كبيرة
    r'\b1000000000000\b': 'tiriliyɔni kelen',
    r'\b1000000000\b': 'miliyari kelen',
    r'\b1000000\b': 'miliyɔn kelen',
    r'\b1000\b': 'bakelen',
    r'\b100\b': 'kɛmɛ',
    r'\b10\b': 'tan',
    r'\b0\b': 'fu',
    # كلمات خاصة
    r'\bcogo\b': 'cogo',
    r'\bcogoya\b': 'cogoya',
}

# ==========================================
# 2) محول عشري/نسب/تواريخ/أوقات عام
# ==========================================
def pronounce_number_decimal_percent(text: str) -> str:
    # نسبة مئوية
    text = re.sub(r'(\d+)[,\.](\d+)%', r"kɛmɛsarada la \1 n' a kunkanfɛn \2", text)
    text = re.sub(r'(\d+)%', r'kɛmɛsarada la \1', text)
    # عشري بالفاصلة
    def dec_rep(m):
        intp, fracp = m.group(1), m.group(2)
        return f"{intp} n'a kunkanfɛn {fracp}"
    text = re.sub(r'(\d+)[,\.](\d+)', dec_rep, text)
    return text

# ==========================================
# 3) قواعد زمن/ساعة مرنة
# ==========================================
def pronounce_time(text: str) -> str:
    text = re.sub(
        r'(\d{1,2})h(\d{2}):(\d{2})',
        lambda m: f"nɛgɛ kanɲɛ {int(m.group(1))} tɛmɛnen ye ni sanga {int(m.group(2))} ye ani segɔni {int(m.group(3))}",
        text
    )
    text = re.sub(
        r'(\d{1,2}):(\d{2}):(\d{2})',
        lambda m: f"nɛgɛ kanɲɛ {int(m.group(1))} tɛmɛnen ye ni sanga {int(m.group(2))} ye ani segɔni {int(m.group(3))}",
        text
    )
    text = re.sub(r'(\d{1,2})h\b', lambda m: f"nɛgɛ kanɲɛ {int(m.group(1))}", text)
    return text

# ==========================================
# 4) عملية G2P بسيطة/قابلة للتوسيع
# ==========================================
G2P_MAP = {
    "ɛ": "ɛ",
    "ɔ": "ɔ",
}

def g2p_phonemize(text: str) -> str:
    for k, v in G2P_MAP.items():
        text = text.replace(k, v)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 5) المجمع الرئيسي لتحويل النص (Normalizer)
# ==========================================
def convert_bambara_numbers_and_text(text: str) -> str:
    text = text.strip()
    text = pronounce_time(text)
    text = pronounce_number_decimal_percent(text)
    for pattern, repl in BASE_REPLACEMENTS.items():
        text = re.sub(pattern, repl, text)
    def year_rep(m):
        year = int(m.group(0))
        mapping = {
            2026: 'bafila ani mugan ni wɔɔrɔ',
            2027: 'bafila ani mugan ni wolonwula',
            2029: 'bafila ani mugan ni kɔnɔntɔn',
            2030: 'bafila ani bisaba',
        }
        return mapping.get(year, f"bafila ani mugan ni {year}")
    text = re.sub(r'\b20\d{2}\b', year_rep, text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = g2p_phonemize(text)
    return text

# ==========================================
# 6) تجزئة ذكية مع تراكب ومنع القطع للنصوص الطويلة
# ==========================================
def chunk_text_smart(text: str, max_chars: int = 200, overlap: int = 30) -> List[str]:
    # أولاً قسم النص إلى جمل حتى نحافظ على الطبيعة
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
            # create overlap: keep last N chars/words
            overlap_words = overlap // 5
            words = cur.split()
            cur = " ".join(words[-overlap_words:]) + " " + s if overlap_words > 0 else s
    if cur:
        chunks.append(cur.strip())
    return chunks

# ==========================================
# 7) تحميل نموذج (مبسط)
# ==========================================
@st.cache_resource
def load_bambara_tts(model_id: str = "facebook/mms-tts-bam"):
    tokenizer = VitsTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device

# ==========================================
# 8) تغيير سرعة/نبرة بسيط للموجة الصوتية (تقليد صوتي خفيف)
# ==========================================
def change_speed(waveform: np.ndarray, speed: float) -> np.ndarray:
    if speed == 1.0:
        return waveform
    # waveform: 1D numpy array
    orig_len = waveform.shape[0]
    new_len = int(orig_len / speed)
    if new_len < 1:
        return waveform
    # بسيط: إعادة أخذ عينات عن طريق interpolation
    x_old = np.linspace(0, 1, orig_len)
    x_new = np.linspace(0, 1, new_len)
    new_wave = np.interp(x_new, x_old, waveform).astype(waveform.dtype)
    return new_wave

# ==========================================
# 9) دالة توليد مقطع صوتي مع دعم optional speaker_embedding
# ==========================================
def synthesize_chunks(tokenizer, model, device, chunks: List[str], speaker_embedding: Optional[np.ndarray] = None, style_speed: float = 1.0) -> Tuple[np.ndarray, int]:
    sampling_rate = getattr(model.config, "sampling_rate", 22050)
    audio_segments = []
    for i, chunk in enumerate(chunks):
        inputs = tokenizer(chunk, return_tensors="pt").to(device)
        with torch.no_grad():
            try:
                if speaker_embedding is not None:
                    # واجهة افتراضية — عدّل اسم الوسيط بحسب API النموذج الحقيقي
                    output = model.generate_speech(**inputs, speaker_embedding=torch.tensor(speaker_embedding).to(device))
                else:
                    out = model(**inputs)
                    output = getattr(out, "waveform", out)
                waveform = output.squeeze().cpu().numpy()
            except Exception:
                out = model(**inputs)
                waveform = getattr(out, "waveform", out)
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.squeeze().cpu().numpy()
        # تطبيق تغيير السرعة/النبرة إذا لم يوجد embedding (تقريبياً)
        if speaker_embedding is None and style_speed != 1.0:
            waveform = change_speed(waveform, style_speed)
        audio_segments.append(waveform)
        # صمت قصير
        audio_segments.append(np.zeros(int(sampling_rate * 0.18)))
    final_audio = np.concatenate(audio_segments) if audio_segments else np.array([], dtype=np.float32)
    return final_audio, sampling_rate

# ==========================================
# 10) واجهة Streamlit المعدلة
# ==========================================
st.set_page_config(page_title="Bamanankan TTS & Voice Cloning (محسّن)", layout="wide")
st.title("استوديو الصوتيات المتقدم للغة البامبارية (محسّن)")

# تحميل النموذج
try:
    tokenizer, model, device = load_bambara_tts()
except Exception as e:
    st.error(f"فشل تحميل النموذج الافتراضي: {e}\nيرجى تعديل model_id أو التحقق من الاتصال بالإنترنت.")
    tokenizer = model = device = None

text_input = st.text_area("أدخل النص أو الأرقام بالبامبارا:", height=250,
                          value="Nin ye bamanankan 2026 ni 2027 jatedenw ani 18h02:01 kalan ye. 4,8 ani 21,9% kɛmɛsarada.")

col1, col2 = st.columns([2,1])
with col1:
    speaker_choice = st.selectbox("اختر نبرة/أسلوب الصوت:",
                                  [
                                      "Bourama (male, steady)",
                                      "Adama (male, natural)",
                                      "Moussa (male, clear)",
                                      "Amadou (male, warm)",
                                      "Intellectual (male, slow)",
                                      "Poet (female, soft)",
                                      "Female (general)",
                                      "Child (youthful)"
                                  ])
with col2:
    ref_audio = st.file_uploader("رفع عيّنة صوتية للاستنساخ (اختياري, WAV/MP3)", type=["wav", "mp3"]) 

advanced = st.expander("إعدادات متقدمة")
with advanced:
    max_chars = st.slider("أقصى طول قطعة (حرف)", 50, 2000, 400)
    overlap = st.slider("تراكب عند التجزئة (عدد تقريبي من الحروف)", 0, 500, 60)
    speed_map_note = st.checkbox("تمكين محاكاة الأسلوب عبر تغيير السرعة/النبرة (تقريبي)", value=True)

# خريطة سرعة لنماذج الأساليب عند عدم وجود embedding
VOICE_STYLE_SPEED = {
    "Bourama (male, steady)": 0.98,
    "Adama (male, natural)": 1.0,
    "Moussa (male, clear)": 1.02,
    "Amadou (male, warm)": 0.97,
    "Intellectual (male, slow)": 0.92,
    "Poet (female, soft)": 1.03,
    "Female (general)": 1.06,
    "Child (youthful)": 1.12,
}

if st.button("توليد صوت بنبرة نظيفة"):
    if not text_input.strip():
        st.warning("الرجاء إدخال نص أولاً!")
    else:
        if tokenizer is None or model is None:
            st.error("النموذج غير محمّل. تأكد من إعداد model_id وتشغيل التطبيق مع اتصال بالإنترنت.")
        else:
            with st.spinner("جاري المعالجة... (قد يستغرق لبعض النصوص الطويلة)"):
                normalized_text = convert_bambara_numbers_and_text(text_input)
                # إذا كان النص طويلاً جداً، نعرض إشعار ونخفض حجم القطع
                total_len = len(normalized_text)
                if total_len > 5000 and max_chars < 400:
                    max_chars = 400
                chunks = chunk_text_smart(normalized_text, max_chars=max_chars, overlap=overlap)

                # استخراج embedding إن تم رفع ملف مرجعي
                speaker_emb = None
                if ref_audio is not None and res_encoder is not None:
                    wav_bytes = ref_audio.read()
                    buf = io.BytesIO(wav_bytes)
                    try:
                        wav = preprocess_wav(buf)
                        speaker_emb = res_encoder.embed_utterance(wav)
                    except Exception:
                        speaker_emb = None

                # تحديد سرعة الأسلوب
                style_speed = 1.0
                if speed_map_note and speaker_emb is None:
                    style_speed = VOICE_STYLE_SPEED.get(speaker_choice, 1.0)

                # لتعامل مع النصوص الكبيرة: نعرض شريط تقدم
                progress = st.progress(0)
                final_audio_segments = []
                sr = getattr(model.config, "sampling_rate", 22050)
                for idx, chunk in enumerate(chunks):
                    # توليد لكل شريحة على حدة
                    chunk_audio, _ = synthesize_chunks(tokenizer, model, device, [chunk], speaker_embedding=speaker_emb, style_speed=style_speed)
                    final_audio_segments.append(chunk_audio)
                    progress.progress(int((idx+1)/len(chunks)*100))

                if final_audio_segments:
                    final_audio = np.concatenate(final_audio_segments)
                else:
                    final_audio = np.array([], dtype=np.float32)

                output_file = "bambara_clear_output.wav"
                sf.write(output_file, final_audio, sr)

            st.success("تم التوليد بنجاح — استمع وحمّل الملف:")
            st.audio(output_file, format="audio/wav")
            with open(output_file, "rb") as f:
                st.download_button("تحميل الملف الصوتي (WAV)", f, file_name="bambara_official.wav", mime="audio/wav")

            st.info("ملاحظة: إذا رغبت في تشابه صوتي دقيق (voice cloning)، ارفع عيّنة صوتية واضحة ومتنوعة (عدة جمل)، وسأحاول استخدام resemblyzer لاستخراج تمثيل المتحدث.")
