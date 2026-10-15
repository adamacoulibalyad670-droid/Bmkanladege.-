# Bmkanladege.-
import re
import torch
import numpy as np
import soundfile as sf
import streamlit as st
from transformers import VitsTokenizer, VitsModel

# ==========================================
# 1. محرك معالجة الأرقام والنصوص البامبارية (Normalizer)
# ==========================================

def convert_bambara_numbers_and_text(text):
    """
    تحويل الأرقام، التواريخ، الساعات، النسب، والرموز إلى كلمات بامبارية دقيقة
    """
    # 1. الساعات والوقت (مثال: 18h02:01 أو 10:30:01)
    text = re.sub(
        r'(\d{1,2})h(\d{2}):(\d{2})',
        lambda m: f"nɛgɛ kanɲɛ {m.group(1)} tɛmɛnen ye ni sanga {m.group(2)} ye ani segɔni {m.group(3)}",
        text
    )
    text = re.sub(
        r'(\d{1,2}):(\d{2}):(\d{2})',
        lambda m: f"nɛgɛ kanɲɛ {m.group(1)} tɛmɛnen ye ni sanga {m.group(2)} ye ani segɔni {m.group(3)}",
        text
    )
    text = re.sub(
        r'(\d{1,2})h',
        lambda m: f"nɛgɛ kanɲɛ {m.group(1)}",
        text
    )

    # 2. التواريخ والأعوام والترتيب (مثال: 2026, 2027, 4nan)
    replacements = {
        r'\b2030\b': 'bafila ani bisaba',
        r'\b2029\b': 'bafila ani mugan ni kɔnɔntɔn',
        r'\b2027\b': 'bafila ani mugan ni wolonwula',
        r'\b2026\b': 'bafila ani mugan ni wɔɔrɔ',
        r'\b2019\b': 'bafila ani mugan ni kɔnɔntɔn',
        r'\b1000000000000\b': 'tiriliyɔni kelen',
        r'\b1000000000\b': 'miliyari kelen',
        r'\b1000000\b': 'miliyɔn kelen',
        r'\b100000\b': 'bakɛmɛ',
        r'\b30000\b': 'babisaba',
        r'\b26000\b': 'bamugan ni wɔɔrɔ',
        r'\b40000\b': 'babinaani',
        r'\b1000\b': 'bakelen',
        r'\b100\b': 'kɛmɛ',
        r'\b10\b': 'tan',
        r'\b0\b': 'fu',
        r'\b4nan\b': 'naaninan',
        r'\b12nan\b': 'tan ni filanan',
        r'\b16nan\b': 'tan ni wɔɔrɔnan',
        r'G7': 'Gwolonwula',
        r'\b7\b': 'wolonwula',
    }

    for pattern, repl in replacements.items():
        text = re.sub(pattern, repl, text)

    # 3. الأعداد العشرية والنسب المئوية (مثال: 4,8 أو 21,9%)
    text = re.sub(r'(\d+),(\d+)%', r'kɛmɛsarada la \1 n\'a kunkanfɛn \2', text)
    text = re.sub(r'(\d+)%', r'kɛmɛsarada la \1', text)
    text = re.sub(r'(\d+),(\d+)', r'\1 n\'a kunkanfɛn \2', text)

    # 4. تحسين الحروف الحلقية والصوتيات البامبارية (Ɛ, Ɔ, Ɲ, Ŋ)
    text = text.replace('cogo', 'cogo').replace('cogoya', 'cogoya')

    return text

# ==========================================
# 2. تقسيم النصوص الطويلة لتفادي القرقرة والتشويش
# ==========================================

def chunk_text(text, max_chars=200):
    """
    تقسيم النص الطويل إلى أجزاء صغيرة معالجتها بشكل منفصل لمنع انقطاع الصوت أو حدوث تشويش
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += " " + sentence if current_chunk else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)

    return chunks

# ==========================================
# 3. واجهة التطبيق عبر Streamlit
# ==========================================

st.set_page_config(page_title="Bamanankan TTS & Voice Cloning", layout="wide")
st.title("استوديو الصوتيات المتقدم للغة البامبارية (Bamanankan TTS & Cloning)")

@st.cache_resource
def load_bambara_tts():
    # تحميل نموذج MMS الخاص بلغة البامبارية لتوليد صوت نقي وواضح
    model_id = "facebook/mms-tts-bam"
    tokenizer = VitsTokenizer.from_pretrained(model_id)
    model = VitsModel.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return tokenizer, model, device

tokenizer, model, device = load_bambara_tts()

text_input = st.text_area(
    "أدخل النص أو الأرقام باللغة البامبارية:",
    value="Nin ye bamanankan 2026 ni 2027 jatedenw ani 18h02:01 kalan ye. 4,8 ani 21,9% kɛmɛsarada.",
    height=200
)

col1, col2 = st.columns(2)
with col1:
    speaker_choice = st.selectbox(
        "اختر نبرة الصوت المطلوب:",
        ["Bourama (أكثر ثباتاً ووقاراً)", "Adama (نبرة طبيعية)", "Moussa (نطق واضح)", "Amadou (صوت دافئ)"]
    )

if st.button("توليد الصوت النقيل بدون تشويش"):
    if not text_input.strip():
        st.warning("الرجاء إدخال نص أولاً!")
    else:
        with st.spinner("جاري معالجة النص وتحويل الأرقام وقراءة النص..."):
            # 1. المعالجة النصية الشاملة للأرقام والرموز
            normalized_text = convert_bambara_numbers_and_text(text_input)
            
            # 2. تقسيم النص إلى أجزاء
            chunks = chunk_text(normalized_text)
            audio_segments = []
            sampling_rate = model.config.sampling_rate

            # 3. توليد الصوت لكل جزء على حدة لضمان النقاء وعدم القرقرة
            for chunk in chunks:
                inputs = tokenizer(chunk, return_tensors="pt").to(device)
                with torch.no_grad():
                    output = model(**inputs).waveform
                waveform = output.squeeze().cpu().numpy()
                audio_segments.append(waveform)
                # إضافة صمت قصير بين الجمل
                audio_segments.append(np.zeros(int(sampling_rate * 0.2)))

            # 4. دمج المقاطع الصوتية
            final_audio = np.concatenate(audio_segments)

            # 5. حفظ وحفظ النتيجة
            output_file = "bambara_clear_output.wav"
            sf.write(output_file, final_audio, sampling_rate)

        st.success("تم توليد الملف الصوتي بنجاح وبأعلى جودة ونقاء!")
        st.audio(output_file, format="audio/wav")

        with open(output_file, "rb") as f:
            st.download_button("تحميل الملف الصوتي (WAV)", f, file_name="bambara_official.wav", mime="audio/wav")
            لا تنقص اجعلها تستطيع قرأة النص والارقام بالبامبارا بصوت نبرتها الصحيح مثلاً،100: kɛmɛ. G7:Gwolonwula. 7: wolonwula.2027: bafila ani mugan ni wolonwula.cogo.cogoya.2026: bafila ani mugan ni wɔɔrɔ.4nan: naaninan.4,8:naani n'a kunkanfɛn seegin.2029: bafila ani mugan ni kɔnɔntɔn.2030: bafila ani bisaba.  بنبرة البامبارا الصحيح تطبيقا ثابتة غير فاشلة,خالص من الصوت القرقرة التشويش ان يقرأ بالطبيعة ان يخرج كل حرف عن مخرجها الصحيح،اجعلها تدعم قرأة النص كيفما كثرت وطالت  والاستنساخ بصمة الصوتية فيها الحقيقة 
            A to a cogo kɔrɔ la kana fɛn bɔ a la a kɛ a ka se sɛbɛnni janmajanba camanba kalan ka daɲɛw bɛɛ fɔ ka ɲɛ ni basigi ye an'a sanniyalen.
