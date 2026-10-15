# Bmkanladege.-

مشروع لتوليد ونطق النصوص باللغة البامبارية (Bamanankan TTS) مع تحسينات لتحويل الأرقام والتواريخ ونسب القراءة ودعم استنساخ الصوت (voice cloning).

المحتويات الرئيسية:
- normalizer: تحويل الأرقام/التواريخ/النسب إلى نطق بامبارا دقيق.
- app.py: تطبيق Streamlit محسّن لتوليد الصوت مع دعم استنساخ الصوت (تمت إضافته بالفعل).

---

## مثال سريع: Bamanankan mankanw code

سريعاً لتجربة نموذج جاهز (مثال):

التثبيت المطلوب (أساسيات):

```bash
pip install transformers torch soundfile
```

مثال كود بسيط لاستخدام نموذج جاهز (sudoping01/bambara-tts):

```python
from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf

# تحميل النموذج (مرة واحدة فقط، بعدين يشتغل offline)
print("جاري تحميل نموذج البامبارا...")
model = VitsModel.from_pretrained("sudoping01/bambara-tts")
tokenizer = AutoTokenizer.from_pretrained("sudoping01/bambara-tts")

def bambara_tts(text, output_file="bambara.wav"):
    """
    يحول نص بامبارا إلى صوت
    """
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        output = model(**inputs).waveform
    
    waveform = output.squeeze().cpu().numpy()
    sample_rate = model.config.sampling_rate  # مثال: 16000
    
    sf.write(output_file, waveform, sample_rate)
    print(f"✅ تم حفظ الصوت: {output_file}")
    return output_file

# ==================== جرب هنا ====================
text = "Aw ni ce. I ka kɛnɛ wa?"   # مرحبا، كيف حالك؟

bambara_tts(text, "hello.wav")

text = "النص اللي تبيه بالبامبارا هنا"
bambara_tts(text, "اسم_الملف.wav")
```

ملاحظة: هذا مثال بسيط لتشغيل نموذج جاهز. قد يختلف اسم النموذج أو توفره حسب المستودع (Hugging Face). اذا كان النموذج متاحًا، سيتم تحميله لأول مرة ثم يعمل محلياً (offline) لاحقاً.

---

## استنساخ الصوت (Voice Cloning)

لا تنس: إذا أردت استنساخ بصمة صوتية حقيقية (voice cloning) بدقة عالية، فعليك التأكد من الآتي:

- استخدام نموذج TTS متعدد المتكلمين (multi-speaker) يدعم conditioning على speaker embeddings أو transfer learning.
- استخراج تمثيل المتحدث من عيّنة صوتية (مثلاً باستخدام resemblyzer أو encoder مناسب) ثم تمريره للنموذج أثناء التوليد.
- قد تحتاج إلى Fine-tuning للنموذج على بيانات المتحدث المستهدف للحصول على تشابه صوتي مرتفع.

في المستودع أضفت ملف `app.py` (Streamlit) مع دعم اختياري لـ resemblyzer لاستخراج speaker embeddings وواجهة رفع عيّنة صوتية للاختبار.

---

## تشغيل تطبيق Streamlit (app.py)

نصائح للتشغيل محلياً:

1. استنخِب/ادخل مجلد المشروع:
   - git clone https://github.com/adamacoulibalyad670-droid/Bmkanladege.-.git
   - cd Bmkanladege.-

2. أنشئ بيئة افتراضية وثبت الحزم:
   - python -m venv .venv && source .venv/bin/activate  (أو على ويندوز: `.venv\Scripts\activate`)
   - pip install --upgrade pip
   - pip install -r requirements.txt  # أو نفّذ: pip install streamlit transformers soundfile numpy torch resemblyzer pyloudnorm noisereduce

3. شغّل التطبيق:
   - streamlit run app.py

ملف `app.py` يحتوي على تحسينات: تجزئة ذكية، تحويلات نطق (normalizer)، G2P مبسّط، ودعم استنساخ صوتي اختياري.

---

## اقتراح: ملف متطلبات

أستطيع إضافة ملف `requirements.txt` تلقائياً يحتوي على الحزم التالية إن رغبت:

```
streamlit
transformers
torch
soundfile
numpy
resemblyzer
pyloudnorm
noisereduce
```

أخبرني إن أردت إضافة `requirements.txt` أو أي تعديل آخر على README أو ضبط `model_id` داخل `app.py` ليتوافق مع نموذج فعلي مثل `sudoping01/bambara-tts`.
