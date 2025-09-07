# gemini_helper.py
import os
import json
import google.generativeai as genai

def _configure_gemini():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set.")
    genai.configure(api_key=key)

def analyze_with_gemini(payload: dict, model_name: str = "gemini-1.5-flash", temperature: float = 0.5) -> str:
    _configure_gemini()
    model = genai.GenerativeModel(model_name)
    json_blob = json.dumps(payload, ensure_ascii=False, indent=2)
    prompt = f"""
    أنت محلل مراهنات رياضية خبير. حلل بيانات JSON التالية وقدم تقريرًا موجزًا وواضحًا باللغة العربية.

    البيانات:
    ```json
    {json_blob}
    ```

    المطلوب:
    1.  قدم نظرة عامة على المباراة بناءً على الاحتمالات العادلة (من هو المرشح؟).
    2.  إذا كانت هناك اقتراحات من "كيلي" (kelly)، اشرح فرصة القيمة (Value) بوضوح.
    3.  قدم توصية واحدة رئيسية بناءً على أكبر "أفضلية" (edge).
    4.  اختتم بفقرة إخلاء مسؤولية.
    استخدم تنسيق Markdown وعناوين وإيموجيز لجعل التقرير سهل القراءة.
    """
    response = model.generate_content(prompt, generation_config={"temperature": temperature})
    return response.text
