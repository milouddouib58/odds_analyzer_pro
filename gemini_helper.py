# gemini_helper.py
import os
import json
import google.generativeai as genai

def _configure_gemini():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=key)

def analyze_with_gemini(payload: dict, model_name: str = "gemini-1.5-flash", temperature: float = 0.6) -> str:
    _configure_gemini()
    model = genai.GenerativeModel(model_name)
    json_blob = json.dumps(payload, ensure_ascii=False, indent=2)
    prompt = f"""
    أنت خبير استراتيجي في المراهنات الرياضية. مهمتك هي مقارنة تحليلين مختلفين وتقديم استشارة نهائية.

    ستصلك بيانات JSON تحتوي على تحليلين:
    1) market_analysis: تحليل مبني على أسعار السوق (الأموال).
    2) statistical_analysis: تحليل مبني على نموذج بواسون الإحصائي (الأداء).

    مهمتك:
    - اشرح موقف كل تحليل.
    - حدد نقاط الاتفاق/الاختلاف.
    - قدم نصيحة استراتيجية واضحة.
    - اختم برأي نهائي موجز.

    البيانات:
    ```json
    {json_blob}
    ```
    """
    resp = model.generate_content(prompt, generation_config={"temperature": temperature})
    text = getattr(resp, "text", None) or ""
    if not text.strip():
        return "لم أتمكن من توليد التحليل حالياً. حاول مجدداً بعد قليل."
    return text
