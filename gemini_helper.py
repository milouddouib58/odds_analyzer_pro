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
    1.  `market_analysis`: تحليل مبني على أسعار السوق (الأموال).
    2.  `statistical_analysis`: تحليل مبني على نموذج بواسون الإحصائي (الأداء).

    مهمتك:
    1.  **اشرح الموقف:** لخص رأي كل محلل. "محلل السوق يرى..."، "بينما المحلل الإحصائي يتوقع...".
    2.  **ابحث عن الاتفاق أو الاختلاف:** هل يتفق المحللان؟ إذا نعم، فهذه إشارة قوية. إذا اختلفا، فهذه إشارة للحذر.
    3.  **قدم نصيحة استراتيجية:** بناءً على الموقف، قدم نصيحة واضحة.
    4.  **أعط رأيك النهائي:** قدم خلاصتك ورأيك الأخير.
    
    البيانات:
    ```json
    {json_blob}
    ```
    """
    response = model.generate_content(prompt, generation_config={"temperature": temperature})
    return response.text
