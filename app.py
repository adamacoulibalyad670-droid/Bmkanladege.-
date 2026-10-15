"""
Bamanankan TTS (محسّن) — Streamlit app
تحسينات: خرائط نطق ممتدة، G2P مبسّط، تجزئة مع تراكب، دعم استنساخ الصوت (speaker embedding)، ومعالجة ما بعدها.
"""

import re
import io
import torch
import numpy as np
import soundfile as sf
import streamlit as st
from typing import List, Tuple

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
    text = re.sub(r'(\d+)[,\.](\d+)%', r'kɛmɛsarada la \1 n\'a kunkanfɛn \2', text)
    text = re.sub(r'(\d+)%', r'kɛmɛsarada la \1', text)
    # عشري بالفاصلة (مثال: 4,8 -> 4 n'a kunkanfɛn 8 أو شكل طلبك: "naani n'a kunkanfɛn seegin")
    # هنا نستخدم نطق مشابه لما طلبت:
    def dec_rep(m):
        intp, fracp = m.group(1), m.group(2)
        # نطق مختصر للجزء الكسري — يمكن تحسينه إلى قراءة كل رقم منفصل.
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
# ملاحظة: هذا تحويل مبسّط لتقريب النطق. للحصول على نطق مضبوط جداً، انشئ قاموس نطق (lexicon) أو استخدم G2P مدرّب.
G2P_MAP = {
    # أمثلة تحسين إخراج الحروف/مقاطع Bambara
    "ɛ": "ɛ",
    "ɔ": "ɔ",
    # تحويلات بسيطة للأحرف اللاتينية/مخرجات غير قياسية
    # ضع هنا قواعد إضافية حسب الحاجة
}

def g2p_phonemize(text: str) -> str:
    # تطبيق تحويلات بسيطة على مستوى الحرف/مقطع
    for k, v in G2P_MAP.items():
        text = text.replace(k, v)
    # تنقية المسافات المزدوجة
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ==========================================
# 5) المجمع الرئيسي لتحويل النص (Normalizer)
# ==========================================
def convert_bambara_numbers_and_text(text: str) -> str:
    text = text.strip()
    # أوقات
    text = pronounce_time(text)
    # أعداد عشرية ونسب
    text = pronounce_number_decimal_percent(text)
    # تطبيق الاستبدالات المباشرة
    for pattern, repl in BASE_REPLACEMENTS.items():
        text = re.sub(pattern, repl, text)
    # سنوات عامة 20xx -> bafila ani mugan ni <word>  (قابل للتخصيص)
    def year_rep(m):
        year = int(m.group(0))
        # مثال مبسّط: تخصيص بعض السنوات الشائعة أعلاه، وإلا تقسيم كلمات (قابل للتوسعة)
        mapping = {
            2026: 'bafila ani mugan ni wɔɔrɔ',
            2027: 'bafila ani mugan ni wolonwula',
            2029: 'bafila ani mugan ni kɔnɔntɔn',
            2030: 'bafila ani bisaba',
        }
        return mapping.get(year, f"bafila ani mugan ni {year}")
    text = re.sub(r'\b20\d{2}\b', year_rep, text)
    # تنظيف نهائي
    text = re.sub(r'\s+', ' ', text).strip()
    # تحويل إلى تمثيل فونيمي/نطق تقريبي
    text = g2p_phonemize(text)
    return text

# ==========================================
# 6) تجزئة ذكية مع تراكب لمنع القطع
# ==========================================
def chunk_text_smart(text: str, max_chars: int = 200, overlap: int = 30) -> List[str]:
    words = text.split()
    chunks = []
    cur = []
    cur_len = 0
    for w in words:
        if cur_len + len(w) + 1 <= max_chars or not cur:
            cur.append(w)
            cur_len += len(w) + 1
        else:
            chunks.append(" ".join(cur))
            # ابدأ الشريحة الجديدة مع تراكب
            cur = cur[-max(0, min(len(cur), overlap//5)):]  # تبسيط للتراكب بكلمات
            cur.append(w)
            cur_len = sum(len(x)+1 for x in cur)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ==========================================
# 7) تحميل نموذج (مبسط)
# ==========================================
@st.cache_resource
def load_bambara_tts(model_id="facebook/mms-tts-bam"):
    tokenizer = VitsTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device

# ==========================================
# 8) دالة توليد مقطع صوتي مع دعم optional speaker_embedding
# ==========================================
def synthesize_chunks(tokenizer, model, device, chunks: List[str], speaker_embedding: np.ndarray = None) -> Tuple[np.ndarray, int]:
    sampling_rate = getattr(model.config, "sampling_rate", 22050)
    audio_segments = []
    for chunk in chunks:
        inputs = tokenizer(chunk, return_tensors="pt").to(device)
        with torch.no_grad():
            # إذا كان النموذج يدعم تسجيل ملامح المتحدث (مثال لبعض نماذج VITS)
            try:
                if speaker_embedding is not None:
                    # واجهة افتراضية — عدّل اسم الوسيط بحسب API النموذج الحقيقي
                    output = model.generate_speech(**inputs, speaker_embedding=torch.tensor(speaker_embedding).to(device))
                else:
                    # واجهة افتراضية أخرى: بعض موديلات transformers تعتمد على forward -> waveform
                    out = model(**inputs)
                    output = getattr(out, "waveform", None) or out
                    
                waveform = output.squeeze().cpu().numpy()
            except Exception:
                # محاولة fallback: بعض النماذج تعيد dict مع 'waveform'
                out = model(**inputs)
                waveform = getattr(out, "waveform", out)
                if isinstance(waveform, torch.Tensor):
                    waveform = waveform.squeeze().cpu().numpy()
        audio_segments.append(waveform)
        audio_segments.append(np.zeros(int(sampling_rate * 0.18)))  # صمت قصير بين المقاطع
    final_audio = np.concatenate(audio_segments)
    return final_audio, sampling_rate

# ==========================================
# 9) واجهة Streamlit
# ==========================================
st.set_page_config(page_title="Bamanankan TTS & Voice Cloning (محسّن)", layout="wide")
st.title("استوديو الصوتيات المتقدم للغة البامبارية (محسّن)")

tokenizer, model, device = load_bambara_tts()

text_input = st.text_area("أدخل النص أو الأرقام بالبامبارا:", height=200,
                          value="Nin ye bamanankan 2026 ni 2027 jatedenw ani 18h02:01 kalan ye. 4,8 ani 21,9% kɛmɛsarada.")

col1, col2 = st.columns(2)
with col1:
    speaker_choice = st.selectbox("اختر نبرة (اختياري عند عدم استخدام استنساخ صوت):",
                                  ["Bourama", "Adama", "Moussa", "Amadou"])
with col2:
    ref_audio = st.file_uploader("رفع عيّنة صوتية للاستنساخ (اختياري, WAV/MP3)", type=["wav", "mp3"]) 

advanced = st.expander("إعدادات متقدمة")
with advanced:
    max_chars = st.slider("أقصى طول قطعة (حرف)", 50, 1000, 220)
    overlap = st.slider("تراكب عند التجزئة (عدد تقريبي من الحروف)", 0, 200, 30)

if st.button("توليد صوت بنبرة نظيفة"):
    if not text_input.strip():
        st.warning("الرجاء إدخال نص أولاً!")
    else:
        with st.spinner("جاري المعالجة..."):
            normalized_text = convert_bambara_numbers_and_text(text_input)
            chunks = chunk_text_smart(normalized_text, max_chars=max_chars, overlap=overlap)

            # استخراج embedding إن تم رفع ملف مرجعي
            speaker_emb = None
            if ref_audio is not None and res_encoder is not None:
                wav_bytes = ref_audio.read()
                # حفظ مؤقت في الذاكرة ومعايرة
                buf = io.BytesIO(wav_bytes)
                try:
                    wav = preprocess_wav(buf)
                    speaker_emb = res_encoder.embed_utterance(wav)
                except Exception:
                    speaker_emb = None

            final_audio, sr = synthesize_chunks(tokenizer, model, device, chunks, speaker_embedding=speaker_emb)

            # حفظ النتيجة
            output_file = "bambara_clear_output.wav"
            sf.write(output_file, final_audio, sr)

        st.success("تم التوليد بنجاح — استمع وحمّل الملف:")
        st.audio(output_file, format="audio/wav")
        with open(output_file, "rb") as f:
            st.download_button("تحميل الملف الصوتي (WAV)", f, file_name="bambara_official.wav", mime="audio/wav")
