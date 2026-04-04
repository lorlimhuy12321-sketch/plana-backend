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
PINECONE_API_KEY = "pcsk_5q84SC_N75kyYFFVJFtYxFNNTDuFDiFC6cWTnzzPyCHx1pUtMg3Zb3zoPFvGwPKPerMAyr"
GEMINI_API_KEY = "AIzaSyAOhOiFQ8AhqmF7FUvgLSkpprrQJ4msVjE"

pc = Pinecone(api_key=PINECONE_API_KEY)                   
index = pc.Index("plana-ai-db")
ai_client = genai.Client(api_key=GEMINI_API_KEY)

@app.route('/ask', methods=['POST'])
def ask_ai():
    data = request.json
    student_question = data.get('question', '')
    base64_image = data.get('image')

    try:
        # ដោយសារយើងលុប Model ធំចេញ យើងនឹងប្រើសំណួរផ្ទាល់ទៅ Gemini តែម្តង
        # ប៉ុន្តែយើងអាចដាក់ Context ខ្លះៗពី Database បើចង់
        
        prompt = f"""
        អ្នកគឺជាគ្រូបង្រៀនគណិតវិទ្យាដ៏ពូកែម្នាក់នៅកម្ពុជា តំណាងឱ្យស្ថាប័ន PlanA Ai។
        សូមដោះស្រាយលំហាត់ក្នុងរូបភាព ឬសំណួរខាងក្រោមឱ្យបានត្រឹមត្រូវបំផុតតាមកម្មវិធីសិក្សានៅកម្ពុជា។
        បង្ហាញវិធីធ្វើមួយជំហានម្តងៗជាភាសាខ្មែរ និងប្រើ LaTeX សម្រាប់រូបមន្ត។
        
        សំណួរ៖ {student_question}
        """

        contents = [prompt]
        if base64_image:
            clean_base64 = base64_image.split(",")[1]
            contents.append(types.Part.from_bytes(data=base64.b64decode(clean_base64), mime_type='image/jpeg'))

        response = ai_client.models.generate_content(model='gemini-2.0-flash', contents=contents)
        return jsonify({"answer": response.text})

    except Exception as e:
        return jsonify({"answer": f"Backend Error: {str(e)}"})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
