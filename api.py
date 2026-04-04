import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app, origins=["https://beautiful-cannoli-5076d2.netlify.app"])

# --- API KEYS (READ FROM SYSTEM) ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)                    
index = pc.Index("plana-ai-db")

# Initialize Gemini
ai_client = genai.Client(api_key=GEMINI_API_KEY)

@app.route('/ask', methods=['POST'])
def ask_ai():
    try:
        data = request.json
        student_question = data.get('question', '')
        base64_image = data.get('image')
        language = data.get('language', 'km') # ទាញយកភាសាដែលសិស្សជ្រើសរើស (Default គឺខ្មែរ)

        # --- 1. SEARCH PINECONE FOR REFERENCE ---
        best_match = "មិនមានឯកសារយោងទេ" 
        
        search_text = student_question if student_question.strip() else "លំហាត់គណិតវិទ្យាថ្នាក់ទី១២"

        try:
            # 🔴 ចំណុចដែលបានកែ៖ ដូរឈ្មោះ Model ទៅជា 'gemini-embedding-001'
            embedding_response = ai_client.models.embed_content(
                model='gemini-embedding-001', 
                contents=search_text,
                # 💡 បញ្ជាក់៖ ប្រសិនបើអ្នកមិនទាន់បានលុប Database ចាស់ (ដែលប្រើទំហំ 384) ទេ 
                # អ្នកត្រូវតែថែមបន្ទាត់ config ខាងក្រោមនេះ។ 
                # (បើអ្នកបានបង្កើត Database ថ្មីទំហំ 3072 ដូចដែលខ្ញុំបានណែនាំពីសារមុន សូមលុបបន្ទាត់ config នេះចោល)
                config={'output_dimensionality': 384} 
            )
            query_vector = embedding_response.embeddings[0].values
            
            search_results = index.query(
                vector=query_vector, 
                top_k=1, 
                include_metadata=True
            )
            
            if search_results['matches']:
                best_match = search_results['matches'][0]['metadata']['text']
                
        except Exception as pinecone_error:
            print(f"Pinecone Error: {pinecone_error}")

        # --- 2. BUILD THE STRICT PROMPT (BASED ON LANGUAGE) ---
        if language == 'en':
            prompt = f"""
            You are an expert math tutor in Cambodia representing PlanA Ai.
            
            TASK: Solve the math problem asked by the user. If they attached an image, read the math from the image.
            
            CRITICAL RULES FOR METHODOLOGY & FORMAT:
            1. RELEVANCE CHECK: If the REFERENCE DATA is irrelevant to the question, ignore it.
            2. STRICT REFERENCE MATCHING: If relevant, you MUST solve the problem using the EXACT mathematical logic and methodology shown in the REFERENCE DATA. 
               -> Translate the explanation into English, but keep the step-by-step mathematical derivation completely identical to the Cambodian standard shown in the reference.
            3. NO CONVERSATIONAL TEXT: Do not say "Hello", "Let's solve this", etc.
            4. START DIRECTLY: Always start your response with exactly "<b>Solution</b>". Do NOT use asterisks (**).
            5. SHORT BRIDGING WORDS: Use only standard English mathematical bridging words (e.g., We have, Let, Therefore, Since, Because).
            6. FINAL CONCLUSION: Always end your solution with exactly "<b>Therefore,</b> [final answer]." Do NOT use asterisks (**).
            7. MATH FORMATTING: Use LaTeX for ALL math formulas.
            
            REFERENCE DATA (from Pinecone database in Khmer):
            {best_match}
            
            USER QUESTION: 
            {student_question}
            """
        else:
            prompt = f"""
            អ្នកគឺជាគ្រូបង្រៀនគណិតវិទ្យាដ៏ពូកែម្នាក់នៅកម្ពុជា តំណាងឱ្យស្ថាប័ន PlanA Ai។
            
            TASK: Solve the math problem asked by the user. If they attached an image, read the math from the image.
            
            CRITICAL RULES FOR METHODOLOGY & FORMAT:
            1. RELEVANCE CHECK (CRITICAL): First, compare the USER QUESTION to the REFERENCE DATA. 
               - If the REFERENCE DATA is about a completely different math topic, you MUST IGNORE the Reference Data entirely.
               - Only apply the Reference Data if it matches the topic of the User Question.
            2. STRICT REFERENCE MATCHING: If the Reference Data is relevant, you MUST solve the problem using the EXACT mathematical logic shown in it.
            3. NO CONVERSATIONAL TEXT: Do not say "សួស្តី", "ខ្ញុំសូមជួយ", "ជំហានទី១", or give any conversational explanations.
            4. START DIRECTLY: Always start your response with exactly "<b>ដំណោះស្រាយ</b>". Do NOT use asterisks (**).
            5. SHORT BRIDGING WORDS: Use only standard Khmer mathematical bridging words (គេមាន, គេបាន, តាង, នាំឱ្យ, ដោយ, ព្រោះ).
            6. FINAL CONCLUSION: Always end your solution with exactly "<b>ដូចនេះ</b> [ចម្លើយចុងក្រោយ] ។" Do NOT use asterisks (**).
            7. MATH FORMATTING: Use LaTeX for ALL math formulas.
            8. RIGOROUS METHODOLOGY: Solve using standard Cambodian high school methodology.
            
            REFERENCE DATA (from Pinecone database):
            {best_match}
            
            USER QUESTION: 
            {student_question}
            """

        contents = [prompt]
        if base64_image:
            clean_base64 = base64_image.split(",")[1]
            contents.append(types.Part.from_bytes(data=base64.b64decode(clean_base64), mime_type='image/jpeg'))

        response = ai_client.models.generate_content(
            model='gemini-2.5-flash', # The best balance of cost and math accuracy
            contents=contents,
            config=types.GenerateContentConfig(
                max_output_tokens=600,  # Caps the response length to save money
                temperature=0.1         # Super low temperature makes it strictly follow your Pinecone math steps without hallucinating
            )
        )
        return jsonify({"answer": response.text})

    except Exception as e:
        return jsonify({"answer": f"Backend Error: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
