from flask import Flask, render_template, request, redirect, url_for
import os, re, requests
import pdfplumber
from collections import Counter
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

MIN_QUESTIONS = 5

HF_API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-base"
HF_HEADERS = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}

STOP_WORDS = {
    "define","explain","describe","what","why","how","importance",
    "advantages","various","related","difference","differentiate",
    "model","models","concept","used","between"
}

# ---------------- HELPERS ----------------
def get_uploaded_files():
    return sorted(f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(".pdf"))

# ---------------- PDF TEXT ----------------
def extract_text_from_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
    return text

# ---------------- QUESTION CANDIDATES ----------------
def extract_question_candidates(text):
    questions = []

    for line in text.split("\n"):
        line = line.strip()
        if len(line) < 25:
            continue

        if "?" in line or line.lower().startswith(
            ("define","explain","describe","differentiate","compare")
        ):
            line = re.sub(r"([a-z])([A-Z])", r"\1 \2", line)
            line = re.sub(r"[•\-–—]", " ", line)
            line = re.sub(r"\s+", " ", line)

            questions.append(line.strip())

    return list(dict.fromkeys(questions))  # remove duplicates

# ---------------- AI CLEANING ----------------
def ai_clean_questions(seed_questions):
    seed = "\n".join(seed_questions[:10])

    prompt = f"""
Generate 5 to 12 SIMPLE, UNIQUE DBMS exam questions.

Rules:
- One question per line
- No repetition
- Clear English
- Short and syllabus-oriented
- End each with '?'

Text:
{seed}
"""

    try:
        r = requests.post(
            HF_API_URL,
            headers=HF_HEADERS,
            json={"inputs": prompt},
            timeout=60
        )
        data = r.json()

        if isinstance(data, list) and "generated_text" in data[0]:
            raw = data[0]["generated_text"]

            lines = re.split(r"\n|\?\s*", raw)
            final = []

            for q in lines:
                q = q.strip()
                if 6 <= len(q.split()) <= 18:
                    q = q.capitalize()
                    if not q.endswith("?"):
                        q += "?"
                    final.append(q)

            final = list(dict.fromkeys(final))  # remove duplicates

            if len(final) >= MIN_QUESTIONS:
                return final   # allow more than 5

    except Exception:
        pass

    return force_questions(seed)

# ---------------- FALLBACK (SAFE 10–12 QUESTIONS) ----------------
def force_questions(_):
    # Predefined syllabus-safe DBMS topics
    topics = [
        "database management system",
        "data model",
        "entity relationship model",
        "normalization",
        "transaction management",
        "concurrency control",
        "database architecture",
        "relational model",
        "indexing",
        "sql",
        "data independence",
        "acid properties"
    ]

    templates = [
        "Define {}?",
        "Explain {}.",
        "Describe {} with an example.",
        "Differentiate {} and relational databases.",
        "Explain the importance of {}."
    ]

    final = []
    used = set()

    for topic in topics:
        for t in templates:
            q = t.format(topic)
            if q not in used:
                used.add(q)
                final.append(q)
            if len(final) >= 12:   # max 12 questions
                return final

    return final[:10]  # safe minimum if fewer

# ---------------- KEYWORDS ----------------
def extract_keywords(questions):
    words = []
    for q in questions:
        for w in re.findall(r"[a-zA-Z]{4,}", q.lower()):
            if w not in STOP_WORDS:
                words.append(w)

    return Counter(words).most_common(10)

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["GET","POST"])
def upload():
    if request.method == "POST":
        files = request.files.getlist("pdfs")
        for f in files:
            if f and f.filename.endswith(".pdf"):
                f.save(os.path.join(
                    UPLOAD_FOLDER, secure_filename(f.filename)
                ))
        return redirect(url_for("upload"))

    return render_template("upload.html", uploaded_files=get_uploaded_files())

@app.route("/analyze", methods=["POST"])
def analyze():
    selected = request.form.getlist("selected_pdfs")
    if not selected:
        return redirect(url_for("upload"))

    candidates = []
    for f in selected:
        path = os.path.join(UPLOAD_FOLDER, f)
        candidates.extend(
            extract_question_candidates(
                extract_text_from_pdf(path)
            )
        )

    final_questions = ai_clean_questions(candidates)
    keywords = extract_keywords(final_questions)

    return render_template(
        "result.html",
        questions=final_questions,
        keywords=keywords
    )

@app.route("/clear", methods=["POST"])
def clear():
    for f in get_uploaded_files():
        os.remove(os.path.join(UPLOAD_FOLDER, f))
    return redirect(url_for("upload"))

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)
