# gemini_helper.py
import os
import json
import google.generativeai as genai

def _configure_gemini():
    """
    تهيئة مكتبة Gemini باستخدام مفتاح API من متغيرات البيئة.
    """
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("GEMINI_API_KEY is not set. Please add it to secrets or the sidebar.")
    genai.configure(api_key=key)

def analyze_with_gemini(payload: dict, model_name: str = "gemini-1.5-flash", temperature: float = 0.6) -> str:
    """
    يحلل بيانات المباراة باستخدام Gemini ويتصرف كرئيس "مجلس الخبراء".
    """
    _configure_gemini()
    model = genai.GenerativeModel(model_name)
    json_blob = json.dumps(payload, ensure_ascii=False, indent=2)
    
    prompt = f"""
    أنت رئيس "مجلس الخبراء" للمراهنات الرياضية. مهمتك هي الاستماع لآراء أربعة خبراء مختلفين، ثم تقديم خلاصة نهائية وتوصية استراتيجية واضحة وموجزة.

    ستصلك آراء الخبراء الأربعة في صيغة JSON:
    1.  `market_analysis`: رأي خبير السوق (مبني على الأموال وأسعار السوق).
    2.  `poisson_analysis`: رأي خبير الأهداف (مبني على متوسط الأهداف المسجلة والمستقبلة).
    3.  `form_analysis`: رأي خبير الأداء الحالي (مبني على نتائج آخر 6 مباريات).
    4.  `xg_analysis`: رأي خبير الأداء النوعي (مبني على جودة الفرص xG).

    مهمتك كـ "رئيس المجلس":
    1.  **عرض الآراء:** لخص بسرعة وبشكل مباشر رأي كل خبير. مثال: "خبير السوق يرشح الفريق A، بينما خبير الأداء الحالي يرشح التعادل...".
    2.  **تحديد مؤشر الثقة:** انظر كم خبيرًا يتفق على نفس النتيجة (فوز المضيف، تعادل، فوز الضيف). هذا هو أهم استنتاج.
        - **اتفاق 4/4 أو 3/4:** صفها بأنها "إشارة ثقة عالية" أو "إجماع قوي".
        - **اتفاق 2/4 (تعارض):** صفها بأنها "إشارة ضعف" أو "انقسام في الآراء"، وانصح بالحذر الشديد.
    3.  **شرح التعارض (إن وجد):** إذا كان هناك تعارض، اشرحه بوضوح. مثال: "على الرغم من أن نتائج الفريق A الأخيرة كانت جيدة (رأي خبير الأداء الحالي)، إلا أن جودة فرصه كانت ضعيفة (رأي خبير xG)، وهذا يبرر التردد في ترشيحه".
    4.  **أعط التوصية النهائية:** بناءً على مؤشر الثقة، قدم توصيتك النهائية ورأيك الأخير بشكل واضح ومباشر. هل الرهان على هذه المباراة فكرة جيدة أم يجب تجنبها؟

    استخدم لغة واثقة ومباشرة. لا تكن مترددًا.

    البيانات:
    ```json
    {json_blob}
    ```
    """
    
    response = model.generate_content(prompt, generation_config={"temperature": temperature})
    return response.text
