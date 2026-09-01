# GLiNER — Classification Signal Extractor

مشروع مستقل لتدريب نموذج GLiNER على نص `issue` فقط. الهدف أن يستخرج النموذج الجزء المفيد للتصنيف مثل اسم النشاط أو نوع العملية، ويتجاهل إشارات الدفع العامة مثل `Apple Pay` و`MADA` و`ATM` عندما لا تكون هي الإشارة التصنيفية المطلوبة.

## محتويات المشروع

- `data/train.jsonl`: عدد 6,563 سجلًا فريدًا للتدريب (أزيل تكرار مطابق واحد).
- `data/validation.jsonl`: عدد 820 سجلًا لضبط النموذج والـ threshold.
- `data/test.jsonl`: عدد 820 سجلًا للاختبار النهائي مرة واحدة.
- `data/all.jsonl`: جميع الأمثلة، للرجوع والمراجعة فقط وليس للتدريب المباشر.
- `data/audit.jsonl`: تفاصيل التدقيق ومصدر كل مثال، وليس مدخلًا للتدريب.
- `config.json`: إعدادات النموذج والتدريب.
- `train.py`: التدريب وحفظ checkpoint كل 250 خطوة.
- `evaluate.py`: حساب Precision وRecall وF1 على validation أو test.
- `predict.py`: تجربة نص issue حقيقي بعد التدريب.
- `validate_data.py`: فحص الصيغة، الحدود، التكرار، وتسرب الأمثلة بين التقسيمات.
- `check_environment.py`: التأكد من رؤية PyTorch لكرت NVIDIA وBF16.

صيغة السجل التدريبي:

```json
{"tokenized_text": ["POS", "PURCHASE", "AT", "ALBAIK"], "ner": [[3, 3, "classification_signal"]]}
```

الرقمان داخل `ner` هما بداية ونهاية الإشارة داخل قائمة الكلمات، والنهاية مشمولة.

## التشغيل على PC بنظام Windows

الطريقة الأنظف هي WSL2 مع Ubuntu. ثبّت أحدث تعريف NVIDIA في Windows، ثم افتح Ubuntu داخل WSL ونفّذ:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd YOUR_REPOSITORY_FOLDER
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

لرفع المشروع أولًا إلى GitHub من جهازك الحالي، أنشئ مستودعًا فارغًا ثم نفّذ من داخل هذا المجلد:

```bash
git init
git add .
git commit -m "Initial GLiNER training project"
git branch -M main
git remote add origin YOUR_GITHUB_REPOSITORY_URL
git push -u origin main
```

أول تثبيت يحتاج إنترنت لتنزيل GLiNER والنموذج الأساسي. بعده افحص الجهاز والداتا:

```bash
python check_environment.py
python validate_data.py
```

يجب أن يظهر كرت RTX وأن تكون `CUDA available: True`. بعدها ابدأ التدريب:

```bash
python train.py
```

الإعداد الافتراضي مناسب كبداية لـ RTX 4070 Ti Super بسعة 16GB: batch size يساوي 4، وعدد الخطوات 3000، وBF16. إذا ظهر خطأ نفاد ذاكرة، غيّر `train_batch_size` و`eval_batch_size` في `config.json` إلى 2 ثم أعد التشغيل.

ستظهر النماذج المحفوظة داخل:

```text
models/classification-signal-v1/checkpoint-*
```

## التقييم الصحيح

ابدأ بالـ validation:

```bash
python evaluate.py --split validation
```

جرّب thresholds مختلفة واختر الأفضل للـ F1 أو حسب تفضيلك بين الدقة والاسترجاع:

```bash
python evaluate.py --split validation --threshold 0.40
python evaluate.py --split validation --threshold 0.50
python evaluate.py --split validation --threshold 0.60
```

بعد اختيار threshold نهائي، عدّله في `config.json` ثم شغّل الاختبار النهائي مرة واحدة:

```bash
python evaluate.py --split test
```

تقارير النتائج والأخطاء تحفظ داخل `reports/`. راجع `mistake_samples` قبل تقرير أن النموذج جاهز.

## تجربة Issue واحد

```bash
python predict.py "YOUR REAL TRANSACTION ISSUE HERE"
```

أو حدد checkpoint بنفسك:

```bash
python predict.py "YOUR ISSUE" --model models/classification-signal-v1/checkpoint-3000
```

## قواعد مهمة

1. لا تدرّب على `test.jsonl` ولا تستخدمه لاختيار threshold.
2. لا تستخدم `all.jsonl` مع التقسيمات الثلاثة، لأنه يحتويها جميعًا وسيصنع تسربًا في التقييم.
3. أفضل checkpoint ليس بالضرورة الأخير؛ قارن نتائج validation بين النقاط المحفوظة.
4. بعد أول تدريب، راجع الأخطاء يدويًا. تحسين الـ labels الخاطئة أهم من زيادة الخطوات عشوائيًا.
5. لا ترفع مجلد `models/` إلى GitHub؛ هو مستبعد من Git تلقائيًا لأنه كبير.
