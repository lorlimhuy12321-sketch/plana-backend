import os
import base64
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
from google import genai
from google.genai import types

app = Flask(__name__)
CORS(app) 

# --- API KEYS ---
# --- API KEYS (READ FROM SYSTEM) ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

pc = Pinecone(api_key=PINECONE_API_KEY)                   
index = pc.Index("plana-ai-db")
ai_client = genai.Client(api_key=GEMINI_API_KEY)

@app.route('/ask', methods=['POST'])
def ask_ai():
    data = request.json
    student_question = data.get('question', '')
    base64_image = data.get('image')

    # --- 1. SEARCH PINECONE FOR REFERENCE ---
        try:
            # (Assuming you have a function that turns the question into a vector)
            # query_vector = get_embedding(student_question) 
            
            # Search Pinecone for the closest match
            # search_results = index.query(vector=query_vector, top_k=1, include_metadata=True)
            
            # Define the variable!
            # best_match = search_results['matches'][0]['metadata']['text']
            
            # FOR NOW: Let's put a dummy string just to test if your code runs without crashing
            best_match = "សន្មតថាឯកសារយោងទទេសិន" 
        except Exception as e:
            best_match = "មិនមានឯកសារយោងទេ"


        # --- 2. BUILD THE STRICT PROMPT ---
        prompt = f"""
        អ្នកគឺជាគ្រូបង្រៀនគណិតវិទ្យាដ៏ពូកែម្នាក់នៅកម្ពុជា តំណាងឱ្យស្ថាប័ន PlanA Ai។
        
        TASK: Solve the math problem asked by the user.
        
        CRITICAL RULES FOR METHODOLOGY & FORMAT:
        1. STRICT REFERENCE MATCHING: Solve the problem using the exact mathematical logic shown in the REFERENCE DATA.
        2. NO CONVERSATIONAL TEXT: Do not say "សួស្តី", "ខ្ញុំសូមជួយ", or give any conversational explanations.
        3. START DIRECTLY: Always start your response with exactly "**ដំណោះស្រាយ**".
        4. SHORT BRIDGING WORDS: Use only standard Khmer mathematical bridging words (គេមាន, គេបាន, តាង, នាំឱ្យ).
        5. FINAL CONCLUSION: Always end your solution with exactly "**ដូចនេះ** [ចម្លើយចុងក្រោយ] ។"
        6. MATH FORMATTING: Use LaTeX for ALL math formulas.
        
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
