import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app) 

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

        # --- 1. SEARCH PINECONE FOR REFERENCE ---
        best_match = "មិនមានឯកសារយោងទេ" # ទុកជាជម្រើសបម្រុង
        
        # បើសិស្សផ្ញើតែរូបភាព អត់មានវាយអក្សរ យើងត្រូវមានពាក្យសម្រាប់ Search
        search_text = student_question if student_question.strip() else "លំហាត់គណិតវិទ្យាថ្នាក់ទី១២"

        try:
            # ជំហាន 1a: បំប្លែងសំណួរទៅជាវ៉ិចទ័រ (Vector Embedding)
            embedding_response = ai_client.models.embed_content(
                model='text-embedding-004', # នេះគឺជា Model សម្រាប់បំប្លែងអក្សរទៅជាលេខ
                contents=search_text
            )
            query_vector = embedding_response.embeddings[0].values
            
            # ជំហាន 1b: ស្វែងរកក្នុង Pinecone យកអាដែលស្រដៀងជាងគេ ១
            search_results = index.query(
                vector=query_vector, 
                top_k=1, 
                include_metadata=True
            )
            
            # ជំហាន 1c: ទាញយកអត្ថបទលំហាត់គំរូចេញមក
            if search_results['matches']:
                # ចំណាំ៖ 'text' នេះត្រូវតែដូចគ្នានឹងឈ្មោះ Field ដែលប្អូនបាន Upload ចូល Pinecone
                best_match = search_results['matches'][0]['metadata']['text']
                
        except Exception as pinecone_error:
            # ប្រសិនបើ Pinecone មានបញ្ហា វានឹង Print ប្រាប់ក្នុង Render តែ App នៅតែដើរធម្មតា
            print(f"Pinecone Error: {pinecone_error}") 

        # --- 2. BUILD THE STRICT PROMPT ---
        prompt = f"""
        អ្នកគឺជាគ្រូបង្រៀនគណិតវិទ្យាដ៏ពូកែម្នាក់នៅកម្ពុជា តំណាងឱ្យស្ថាប័ន PlanA Ai។
        
        TASK: Solve the math problem asked by the user. If they attached an image, read the math from the image.
        
        CRITICAL RULES FOR METHODOLOGY & FORMAT:
        1. NO CONVERSATIONAL TEXT: Do not say "សួស្តី", "ខ្ញុំសូមជួយ", "ជំហានទី១", or give any conversational explanations.
        2. START DIRECTLY: Always start your response with exactly "<b>ដំណោះស្រាយ</b>". Do NOT use asterisks (**).
        3. SHORT BRIDGING WORDS: Use only standard Khmer mathematical bridging words (គេមាន, គេបាន, តាង, នាំឱ្យ, ដោយ, ព្រោះ).
        4. FINAL CONCLUSION: Always end your solution with exactly "<b>ដូចនេះ</b> [ចម្លើយចុងក្រោយ] ។" Do NOT use asterisks (**).
        5. MATH FORMATTING: Use LaTeX for ALL math formulas.
        6. RIGOROUS METHODOLOGY: Solve the problem using the exact mathematical logic shown in the REFERENCE DATA or standard Cambodian high school methodology (like using \\lim for asymptotes). Do NOT use shortcuts.
        
        REFERENCE DATA (from Pinecone database):
        {best_match}
        
        USER QUESTION: 
        {student_question}
        """

        contents = [prompt]
        if base64_image:
            clean_base64 = base64_image.split(",")[1]
            contents.append(types.Part.from_bytes(data=base64.b64decode(clean_base64), mime_type='image/jpeg'))

        response = ai_client.models.generate_content(model='gemini-2.5-flash', contents=contents)
        return jsonify({"answer": response.text})

    except Exception as e:
        return jsonify({"answer": f"Backend Error: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
