# نتائج تجربة GLiNER v2 على GPU

اكتملت 3,000 خطوة بتاريخ 2026-09-07 على RTX 4070 SUPER بذاكرة 12GB. أفضل موديل اختير من validation عند الخطوة 2,000، وهو محفوظ محليًا في `models/classification-signal-v2/best`.

**التدريب مكتمل، لكن الموديل يحتاج معالجة أخطاء ومراجعة بشرية قبل الإنتاج.** نتيجة validation العالية تقيس الاتفاق مع تعليقات v2 المؤقتة. فحص sanity المنفصل كشف أخطاء في حدود الأسماء واستبعاد قنوات الدفع؛ ليس معيارًا بشريًا مستقلاً.

## النتائج المقاسة

| القياس | الأمثلة | Precision | Recall | F1 | تطابق النص كاملًا | استخراج خاطئ من الحالات الفارغة |
|---|---:|---:|---:|---:|---:|---:|
| الموديل الأصلي — validation | 1,468 | 50.00% | 0.12% | 0.24% | 5.31% | 0 / 77 |
| أفضل موديل مدرب — validation | 1,468 | 98.96% | 99.02% | 98.99% | 98.71% | 0 / 77 |
| أفضل موديل مدرب — sanity | 20 | 81.25% | 72.22% | 76.47% | 80.00% | 0 / 3 |

على validation: 1,625 مقطعًا صحيحًا، و17 استخراجًا زائدًا، و16 مقطعًا فائتًا. تطابقت كل المقاطع في 1,449 من 1,468 نصًا. على sanity تطابقت 16 من 20 حالة بالكامل.

قارنا العتبات 0.4 و0.5 و0.6 على validation فقط؛ تساوت المقاييس، فاحتُفظ بالعتبة الأصلية **0.5**. لم يُستخدم test لتقييم هذا الموديل أو ضبط اختيارات هذه التجربة؛ يبقى للاختبار النهائي بعد تثبيت السياسة.

## أخطاء sanity الأربعة

| النص | الهدف المسجّل | تنبؤ الموديل |
|---|---|---|
| `. Credit Profit` | `Credit Profit` | `. Credit Profit` |
| `Online Purchase from Mcdonalds Riyadh Mada Riyadh` | `Mcdonalds` | `Mcdonalds Riyadh Mada` |
| `. Disclaimer Letter Charges` | `Disclaimer Letter Charges` | `[]` |
| `4092013488    : APPLE.COM/BILL ITUNES.COM IE` | `APPLE.COM/BILL + ITUNES.COM` | `APPLE.COM/BILL ITUNES.COM` |

هذه الأخطاء تحتاج مراجعة تغطية الأنماط في بيانات التدريب وتقييمًا بشريًا جديدًا. لا تُنقل حالات sanity نفسها إلى التدريب بهدف رفع نتيجتها، ولا تُعتمد النتيجة الإجمالية وحدها لإثبات جودة البنوك ذات الأمثلة القليلة.

## التجربة الفعلية للنص المختلط

مررنا النص الذي أرسله المستخدم إلى أفضل موديل. في حالته الكاملة مع CITY/MADA/الرسوم، وكذلك عند إرسال الاسم وحده، كان المقطع المستخرج:

```json
["مؤسسة روائع المrwea almaktbat"]
```

النص الكامل موجود في train، لذلك هذه تجربة استعمال وليست دليلًا مستقلًا على التعميم. التفاصيل والثقة والمواقع في `reports/example_predictions.json`.

## الإعداد والتنفيذ

- النموذج الأصلي: `urchade/gliner_multi-v2.1`؛ المدخل issue فقط، والنوع الداخلي `classification_signal`.
- التدريب: 11,606 مثالًا، batch=2، gradient accumulation=4، seed=42، BF16، وحفظ/تقييم كل 250 خطوة.
- البيئة: Python 3.11.16، PyTorch 2.8.0+cu128، GLiNER 0.2.28، Transformers 4.57.6، Accelerate 1.12.0.
- نجحت الاختبارات الـ17 والتحقق من البيانات وحسابات CUDA وBF16 وفحص حفظ حالة العشوائية.
- نجحت تجربة خمس خطوات، مع حفظ optimizer وscheduler وRNG. توقف التشغيل الأول بعد الخطوة 1,919 لسبب غير ظاهر؛ استؤنف من checkpoint-1750 بعد التحقق من تطابق البيانات والإعدادات ثم اكتمل.
- أصلح `train.py` استهلاك حالة العشوائية أثناء تقييم on_save باستخدام `torch.random.fork_rng`؛ هذا يحافظ على تسلسل التدريب عند الاستئناف.
- `best` للاستعمال، و`final` لأوزان الخطوة 3,000. ملفات الاستئناف الحالية: checkpoint-2500 وcheckpoint-2750 وcheckpoint-3000.

## أوامر هذا الجهاز داخل مجلد GLINER الداخلي

مشغّل `reports/run_local.py` يستخدم Python الخاص بالمشروع والكاش المحلي للعمل دون اتصال بـHugging Face:

```bash
.venv/bin/python reports/run_local.py predict.py "ATM Cash Withdrawal"
.venv/bin/python reports/run_local.py predict.py "مؤسسة روائع المrwea almaktbat"
.venv/bin/python reports/run_local.py evaluate.py --split validation
.venv/bin/python reports/run_local.py evaluate.py --split sanity
```

التدريب الحالي مكتمل. لأي تجربة مختلفة استخدم مجلد output جديدًا؛ عند استئناف تجربة انقطعت استخدم نفس بياناتها وإعداداتها وcheckpoint يحتوي حالة optimizer، كما يوضح `PC_HANDOFF.md`.

## الملفات المحلية

- الملخص القابل للقراءة آليًا: `reports/training_summary.json`.
- سياق التشغيل وبصمات المصدر والكود: `reports/pc_run_context.json`.
- المكتبات المثبتة: `reports/pc_requirements_freeze.txt`.
- حالة المراحل المكتملة: `reports/training_job_status.json`.
- خط الأساس: `reports/validation_20260906T192117512069Z.json`.
- التحقق النهائي: `reports/validation_20260907T035758414665Z.json`.
- sanity: `reports/sanity_20260907T035807385129Z.json`.
- اختيار العتبة: `reports/threshold_selection.json`.
- السجلات: `reports/pc_training.log` و`reports/pc_validation.log` و`reports/pc_sanity.log`، مع حفظ سجلات التشغيل المنقطع بأسماء مستقلة.

النماذج والتقارير التشغيلية محلية ومستبعدة من Git. لا يعني وجود هذا الملف أن الأوزان مرفوعة للمستودع.
