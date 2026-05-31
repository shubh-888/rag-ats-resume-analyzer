from dotenv import load_dotenv
load_dotenv()

import os
import json
import fitz
# import google.generativeai as genai
from groq import Groq
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from flask import (
    Flask,
    render_template,
    request,
    send_file
)

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import (
    getSampleStyleSheet
)
# ---------------- GEMINI CONFIG ---------------- #


key = os.getenv("GOOGLE_API_KEY")
key = os.getenv("GROQ_API_KEY")

print("Groq Key Found:", key is not None)

if key:
    print("Using key:", key[:10] + "...")
else:
    print("GROQ_API_KEY not found")
print("Using key:", key[:10] + "...")
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

app = Flask(__name__)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ---------------- PDF EXTRACTION ---------------- #

# def extract_pdf_text(pdf_file):

#     pdf_document = fitz.open(
#         stream=pdf_file.read(),
#         filetype="pdf"
#     )

#     text = ""

#     for page in pdf_document:
#         text += page.get_text()

#     return text

# def generate_pdf(content):

#     pdf_path = "report.pdf"

#     doc = SimpleDocTemplate(
#         pdf_path
#     )

#     styles = getSampleStyleSheet()

#     elements = []

#     elements.append(
#         Paragraph(
#             "ATS Resume Report",
#             styles["Title"]
#         )
#     )

#     elements.append(
#         Spacer(1,20)
#     )

#     elements.append(
#         Paragraph(
#             content.replace(
#                 "\n",
#                 "<br/>"
#             ),
#             styles["BodyText"]
#         )
#     )

#     doc.build(elements)

#     return pdf_path

#     text = ""

#     for page in pdf_document:
#         text += page.get_text()

#     return text

# ---------------- PDF EXTRACTION ---------------- #

def extract_pdf_text(pdf_file):

    pdf_document = fitz.open(
        stream=pdf_file.read(),
        filetype="pdf"
    )

    text = ""

    for page in pdf_document:
        text += page.get_text()

    return text


# ---------------- PDF REPORT ---------------- #

def generate_pdf(content):

    pdf_path = "report.pdf"

    doc = SimpleDocTemplate(
        pdf_path
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "ATS Resume Report",
            styles["Title"]
        )
    )

    elements.append(
        Spacer(1,20)
    )

    elements.append(
        Paragraph(
            str(content).replace(
                "\n",
                "<br/>"
            ),
            styles["BodyText"]
        )
    )

    doc.build(elements)

    return pdf_path

def chunk_text(text):

    chunk_size = 1000
    chunks = []

    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])

    return chunks


def build_faiss_index(chunks):

    embeddings = embedding_model.encode(chunks)

    embeddings = np.array(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    return index


def retrieve_context(
    query,
    chunks,
    index,
    k=5
):

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding,
        dtype="float32"
    )

    distances, indices = index.search(
        query_embedding,
        k
    )

    context = ""

    for idx in indices[0]:
        context += chunks[idx] + "\n\n"

    return context

# ---------------- GEMINI ---------------- #

def get_gemini_response(
    job_description,
    resume_text,
    prompt
    
):

    chunks = chunk_text(
        resume_text
    )

    index = build_faiss_index(
        chunks
    )

    relevant_context = retrieve_context(
        job_description,
        chunks,
        index
    )
    # DEBUGGING
    print("Chunks Created:", len(chunks))
    print("Retrieved Context:")
    print(relevant_context[:500])

    final_prompt = f"""
{prompt}

JOB DESCRIPTION:
{job_description}

RELEVANT RESUME CONTENT:
{relevant_context}
"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        temperature=0.3,
        max_tokens=2048
    )

    return response.choices[0].message.content

def generate_pdf(content):

    pdf_path = "report.pdf"

    doc = SimpleDocTemplate(pdf_path)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "ATS Resume Report",
            styles["Title"]
        )
    )

    elements.append(
        Spacer(1,20)
    )

    elements.append(
        Paragraph(
            str(content).replace("\n","<br/>"),
            styles["BodyText"]
        )
    )

    doc.build(elements)

    return pdf_path


# ---------------- PROMPTS ---------------- #

review_prompt = """
You are an experienced Technical HR Manager.

Review the resume against the provided job description.

Provide:

1. Overall Evaluation
2. Strengths
3. Weaknesses
4. Missing Skills
5. Hiring Recommendation

Be detailed and professional.
"""

ats_prompt = """
Analyze the resume against the job description.

Return ONLY valid JSON.

{
  "ats_score": 0,
  "candidate_status": "",
  "strengths": [],
  "weaknesses": [],
  "missing_skills": [],
  "recommendation": ""
}

Rules:
- Return pure JSON only.
- No markdown.
- No ```json.
- No explanation text.
"""
skill_gap_prompt = """
Analyze the resume against the job description.

Provide:

1. Skills Present
2. Missing Skills
3. Most Important Missing Skills
4. 30-Day Learning Roadmap
5. Recommended Projects
6. Recommended Certifications
7. Career Advice
"""

interview_prompt = """
Generate:

1. 10 Technical Questions
2. 5 HR Questions
3. 5 Coding Questions
4. 5 Project-Based Questions

Focus on missing skills.
"""

resume_optimizer_prompt = """
You are an expert ATS Resume Writer.

Rewrite the resume to maximize ATS compatibility.

Rules:

- Don't invent experience.
- Don't add fake projects.
- Improve Summary.
- Improve Skills.
- Improve Project Descriptions.

Return:

1. Optimized Summary
2. Optimized Skills
3. Optimized Projects
4. ATS Improvement Tips
"""
roadmap_prompt = """
Analyze the resume against the job description.

Provide:

1. Missing Skills
2. 30-Day Learning Roadmap
3. Weekly Plan
4. Recommended Courses
5. Recommended Projects

Make roadmap practical.
"""

# ---------------- ROUTE ---------------- #

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        action = request.form.get("action")

        job_description = request.form.get(
            "job_description"
        )

        resume_file = request.files.get("resume")

        if not resume_file or resume_file.filename == "":
            return render_template(
                "index.html",
                result="❌ Please upload a PDF resume."
            )

        if not job_description or not job_description.strip():
             return render_template(
                "index.html",
                result="❌ Please enter a Job Description."
            )

        resume_text = extract_pdf_text(
            resume_file
        )

        if action == "review":
            result = get_gemini_response(
                job_description,
                resume_text,
                review_prompt
            )
            generate_pdf(str(result))

        elif action == "ats":
            response = get_gemini_response(
                job_description,
                resume_text,
                ats_prompt
            )
            generate_pdf(response)

            print("RAW GEMINI RESPONSE:")
            print(response)

            response = response.strip()
            response = response.replace("```json", "")
            response = response.replace("```", "")
            try:
                result = json.loads(response)
            except Exception:
                result = {
                    "ats_score": 0,
                    "candidate_status": "Parsing Error",
                    "strengths": [],
                    "weaknesses": [],
                    "missing_skills": [],
                    "recommendation": response
                }

        elif action == "gap":
            result = get_gemini_response(
                job_description,
                resume_text,
                skill_gap_prompt
            )
            generate_pdf(str(result))

        elif action == "interview":
            result = get_gemini_response(
                job_description,
                resume_text,
                interview_prompt
            )
            generate_pdf(str(result))
            
        elif action == "optimize":
            result = get_gemini_response(
                job_description,
                resume_text,
                resume_optimizer_prompt
            )
        elif action == "roadmap":
            result = get_gemini_response(
                job_description,
                resume_text,
                roadmap_prompt
            )

    return render_template(
        "index.html",
        result=result,
        is_json=isinstance(result, dict)
    )

@app.route("/download")
def download():
    return send_file(
        "report.pdf",
        as_attachment=True
    )

# ---------------- RUN ---------------- #

# if __name__ == "__main__":
#     app.run(
#         debug=True
#     )
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)