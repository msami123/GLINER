# GLiNER — فلتر إشارات التصنيف

المدخل: **نص `issue` فقط**. المخرج: اسم تاجر/نشاط أو إجراء مالي ظاهر في النص، مثل `Cash Withdrawal`. يتعلم تجاهل `Apple Pay` و`MADA` و`ATM` كقنوات دفع. هذا المشروع لا يدرب مصنف المصروفات حاليًا.

ابدأ من [PC_HANDOFF.md](PC_HANDOFF.md): ملخص المحادثة، ما تغير، خطة العمل، وخطوات PC والاستئناف.

اكتملت أول تجربة GPU من 3,000 خطوة بتاريخ 2026-09-07. الموديل المختار موجود محليًا في `models/classification-signal-v2/best`. راجع [نتائج التدريب وحدودها](TRAINING_RESULTS.md) قبل الاستخدام؛ توجد أخطاء مهمة في فحص sanity رغم ارتفاع نتيجة validation.

## النسخة الجاهزة

| القسم | عدد الأمثلة | أمثلة هدفها فارغ |
|---|---:|---:|
| التدريب | 11,606 | 414 |
| التحقق | 1,468 | 77 |
| الاختبار | 1,471 | 54 |
| فحص sanity منفصل | 20 | 3 |

البيانات الجديدة في `data/v2/`. المجموع الأساسي 14,545، إضافة إلى 20 حالة sanity و1,500 حالة غير معلّمة للمراجعة. مررنا على 387,408 issue وعلى 20,676 صفًا مفصلًا؛ لم نحول كل البيانات إلى تعليقات آلية غير موثوقة. راجع [سياسة التعليق](ANNOTATION_POLICY.md) و[إحصاءات المصدر](data/v2/summary.json).

أمثلة حقيقية:

```json
{"issue":"Online Purchase from Mcdonalds Riyadh Mada Riyadh","signals":["Mcdonalds"]}
{"issue":"ATM Cash Withdrawal","signals":["Cash Withdrawal"]}
{"issue":"CITY:DAMMAM مدى:**9434 11588850 DAMMAM","signals":[]}
```

`signals` أعلاه شرح للمطلوب، وليست مخرجات موديل مدرّب. التمثيل الذي يقرأه تدريب GLiNER للمثال الثاني:

```json
{"tokenized_text":["ATM","Cash","Withdrawal"],"ner":[[1,2,"classification_signal"]],"ner_labels":["classification_signal"]}
```

نوع واحد فقط مطلوب داخليًا لتعليم موضع المقاطع. لا تحتاج لتمرير بنك أو مبلغ أو اسم تاجر وقت الاستخدام.

## بعد تجهيز بيئة CUDA كما في ملف التسليم

```bash
python check_environment.py
python validate_data.py
python -m unittest discover -s tests -v
python evaluate.py --base-model --split validation
python train.py --smoke-test
python train.py
```

للاستئناف بنفس البيانات والإعدادات:

```bash
python train.py --resume latest
```

للتقييم والتجربة بعد التدريب:

```bash
python evaluate.py --split validation
python evaluate.py --split sanity
python predict.py "ATM Cash Withdrawal"
```

يختار البرنامج أفضل حفظ بحسب F1 على validation بالنص الأصلي. الملفات `models/.../checkpoint-*` تحفظ حالة الاستئناف، و`models/.../best` للاستعمال. لا تدرّب على test ولا تكرر ضبط القرارات عليه.

## حدود ما تم التحقق منه

اختبارات تجهيز البيانات، صحة المقاطع، منع التكرار والتداخل اجتازت الفحص محليًا، ونجحت تجربة التدريب القصيرة ثم اكتمل تدريب GPU والتحقق من ملفات الاستئناف. معظم التعليقات بقواعد وتحتاج تقييمًا بشريًا مستقلًا قبل الإنتاج؛ نتيجة validation المرتفعة لا تعني اجتياز كل الحالات المهمة. التفاصيل في `TRAINING_RESULTS.md`.
