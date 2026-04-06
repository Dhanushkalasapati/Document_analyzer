import streamlit as st
import json
import math
import fitz
import threading
from pathlib import Path
from google import genai

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Analysis Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0f0f1a; }
  .block-container { padding: 2rem 2rem 4rem; max-width: 1100px; }

  .hero { text-align:center; padding: 2rem 0 1rem; }
  .hero h1 { font-size: 2.4rem; font-weight: 700; color: #ffffff; margin-bottom: 0.3rem; }
  .hero p  { font-size: 1rem; color: #888; margin: 0; }

  .card {
    background: #1a1a2e; border: 1px solid #2a2a4a;
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
  }
  .card-title { font-size: 1rem; font-weight: 600; color: #fff; margin-bottom: 0.3rem; }
  .card-desc  { font-size: 0.88rem; color: #aaa; line-height: 1.6; }

  .topic-card {
    border-radius: 12px; padding: 1rem 1.2rem;
    margin-bottom: 0.7rem; border: 1px solid rgba(255,255,255,0.08);
    cursor: pointer; transition: all 0.2s;
  }
  .topic-title { font-size: 1rem; font-weight: 600; margin-bottom: 0.3rem; }
  .topic-desc  { font-size: 0.87rem; line-height: 1.6; }

  .summary-box {
    background: #1a1a2e; border: 1px solid #2a2a4a;
    border-radius: 12px; padding: 1.4rem 1.6rem;
    font-size: 0.95rem; line-height: 1.8; color: #ddd;
  }

  .flashcard {
    background: #1a1a2e; border: 2px solid #4285F4;
    border-radius: 16px; padding: 2rem;
    text-align: center; min-height: 180px;
    display: flex; flex-direction: column; justify-content: center;
  }
  .fc-question { font-size: 1.1rem; font-weight: 600; color: #fff; margin-bottom: 1rem; }
  .fc-answer   { font-size: 0.95rem; color: #4285F4; margin-top: 1rem; }

  .stButton > button {
    background: #4285F4 !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; padding: 0.6rem 1.4rem !important;
    width: 100%;
  }
  .stButton > button:hover { background: #2a6dd9 !important; }

  .stTextInput > div > input {
    background: #1a1a2e !important; color: #fff !important;
    border: 1px solid #2a2a4a !important; border-radius: 8px !important;
  }

  .badge {
    display: inline-block; padding: 3px 10px;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
  }

  .mindmap-container {
    background: #0d0d1a; border-radius: 14px;
    padding: 1rem; overflow: auto;
  }

  div[data-testid="stSidebar"] { background: #12122a !important; }
  div[data-testid="stSidebar"] * { color: #ccc; }

  h2, h3 { color: #fff !important; }
  .stTabs [data-baseweb="tab"] { color: #888 !important; font-weight: 500; }
  .stTabs [aria-selected="true"] { color: #4285F4 !important; border-bottom-color: #4285F4 !important; }
  .stMarkdown p { color: #ccc; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
COLORS = ["#4285F4","#34A853","#EA4335","#FBBC05","#7c6ff7","#4cc9b8","#e0913a","#e05c7a"]
BG     = ["#1a2a4d","#1a3328","#3d1a1a","#3d3110","#2d2850","#103830","#3d2a10","#3d1a24"]

# ── Session state ─────────────────────────────────────────────────────────────
for key in ["analysis","expanded_topics","fc_index","fc_flipped"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "analysis" else ([] if key == "expanded_topics" else 0 if key == "fc_index" else False)

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_text(file_bytes, filename):
    if filename.lower().endswith(".pdf"):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = ""
        for i in range(min(doc.page_count, 50)):
            text += doc[i].get_text() + "\n"
        return text
    return file_bytes.decode("utf-8", errors="ignore")

def call_gemini(api_key, prompt):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    return response.text.strip()

def analyze_document(api_key, text):
    truncated = text[:20000] + ("\n\n[Truncated]" if len(text) > 20000 else "")
    prompt = f"""Analyze the document below. Return ONLY a raw JSON object with no markdown, no backticks.

Structure:
{{
  "summary": "4-6 sentence paragraph summarizing the document.",
  "topics": [
    {{"title": "Topic name", "description": "1-2 sentence explanation.", "color_index": 0}}
  ],
  "mindmap": {{
    "center": "Main subject (max 4 words)",
    "branches": [
      {{"label": "Branch label (2-4 words)", "children": ["child 1", "child 2"]}}
    ]
  }},
  "flashcards": [
    {{"question": "A question about the document?", "answer": "Clear concise answer."}}
  ]
}}

Rules:
- topics: 5-7 items, color_index 0 to 7 assigned sequentially
- mindmap: 4-6 branches, each with 2-4 children, all labels max 4 words
- flashcards: 6-8 cards covering key concepts
- Return ONLY the JSON object

Document:
---
{truncated}
---"""
    raw = call_gemini(api_key, prompt)
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    return json.loads(raw.strip())

def get_topic_detail(api_key, title, desc, summary):
    prompt = f"""The user is studying a document. Give a detailed explanation (5-8 sentences) of this topic.

Topic: {title}
Brief description: {desc}
Document context: {summary}

Write a clear educational explanation with key points."""
    return call_gemini(api_key, prompt)

def render_mindmap(mm):
    branches = mm.get("branches", [])
    center   = mm.get("center", "Document")
    n = len(branches)
    if n == 0: return ""

    W, H = 700, 500
    cx, cy = W//2, H//2
    R_br, R_ch = 130, 90

    svg = f'<svg width="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
    svg += '<rect width="100%" height="100%" fill="#0d0d1a" rx="12"/>'

    for i, branch in enumerate(branches):
        angle = -math.pi/2 + (2*math.pi/n)*i
        bx = cx + R_br * math.cos(angle)
        by = cy + R_br * math.sin(angle)
        col = COLORS[i % len(COLORS)]

        svg += f'<line x1="{cx}" y1="{cy}" x2="{bx}" y2="{by}" stroke="{col}" stroke-width="2" stroke-opacity="0.6"/>'

        label = branch.get("label","")
        bw = max(80, len(label)*8 + 20)
        svg += f'<rect x="{bx-bw//2}" y="{by-14}" width="{bw}" height="28" rx="14" fill="{col}"/>'
        svg += f'<text x="{bx}" y="{by+5}" text-anchor="middle" font-size="11" font-weight="600" fill="white" font-family="Inter,sans-serif">{label}</text>'

        children = branch.get("children", [])
        nc = len(children)
        spread = math.pi * 0.6
        for j, child in enumerate(children):
            ca  = angle - spread/2 + (j*spread/(nc-1) if nc > 1 else 0)
            chx = bx + R_ch * math.cos(ca)
            chy = by + R_ch * math.sin(ca)
            cw  = max(60, len(child)*7 + 16)
            bgc = BG[i % len(BG)]

            svg += f'<line x1="{bx}" y1="{by}" x2="{chx}" y2="{chy}" stroke="{col}" stroke-width="1.2" stroke-dasharray="4 3" stroke-opacity="0.5"/>'
            svg += f'<rect x="{chx-cw//2}" y="{chy-11}" width="{cw}" height="22" rx="11" fill="{bgc}" stroke="{col}" stroke-width="1"/>'
            svg += f'<text x="{chx}" y="{chy+4}" text-anchor="middle" font-size="10" fill="{col}" font-family="Inter,sans-serif">{child}</text>'

    svg += f'<ellipse cx="{cx}" cy="{cy}" rx="70" ry="30" fill="#1a237e" stroke="#4285F4" stroke-width="2"/>'
    svg += f'<text x="{cx}" y="{cy+5}" text-anchor="middle" font-size="12" font-weight="700" fill="white" font-family="Inter,sans-serif">{center}</text>'
    svg += '</svg>'
    return svg

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Document Analysis Agent")
    st.markdown("*Powered by Google Gemini · Free*")
    st.divider()

    api_key = st.text_input("Gemini API Key", type="password",
                             placeholder="AIzaSy...",
                             help="Get free key at aistudio.google.com")
    st.caption("🔑 Free key from [aistudio.google.com](https://aistudio.google.com)")

    uploaded = st.file_uploader("Upload Document", type=["pdf","txt","md"],
                                 help="PDF or plain text, max 10MB")

    analyze = st.button("🔍 Analyze Document", use_container_width=True)

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Enter your free Gemini API key\n2. Upload a PDF or text file\n3. Click Analyze\n4. Get Summary, Topics, Mind Map & Flashcards")

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>📄 Document Analysis Agent</h1>
  <p>Upload any document — get instant AI-powered summary, key topics, mind map & flashcards</p>
</div>
""", unsafe_allow_html=True)

# ── Analysis trigger ──────────────────────────────────────────────────────────
if analyze:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not uploaded:
        st.error("Please upload a document first.")
    else:
        with st.spinner("Analyzing your document with Gemini AI..."):
            try:
                text = extract_text(uploaded.read(), uploaded.name)
                if not text.strip():
                    st.error("No readable text found. Try a text-based PDF.")
                else:
                    st.session_state.analysis = analyze_document(api_key, text)
                    st.session_state.expanded_topics = []
                    st.session_state.fc_index   = 0
                    st.session_state.fc_flipped = False
                    st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Analysis failed: {e}")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.analysis:
    d = st.session_state.analysis
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Summary", "🏷️ Key Topics", "🧠 Mind Map", "🃏 Flashcards"])

    # ── Summary ───────────────────────────────────────────────────────────────
    with tab1:
        st.markdown(f'<div class="summary-box">{d.get("summary","")}</div>', unsafe_allow_html=True)

    # ── Key Topics ────────────────────────────────────────────────────────────
    with tab2:
        st.markdown("*Click any topic to expand a detailed explanation*")
        for i, t in enumerate(d.get("topics",[])):
            ci  = t.get("color_index", i) % len(COLORS)
            col = COLORS[ci]; bg = BG[ci]
            is_expanded = i in st.session_state.expanded_topics
            arrow = "▼" if is_expanded else "▶"

            st.markdown(f"""
            <div class="topic-card" style="background:{bg}; border-color:{col}40;">
              <div class="topic-title" style="color:{col};">{arrow} {t.get('title','')}</div>
              <div class="topic-desc">{t.get('description','')}</div>
            </div>""", unsafe_allow_html=True)

            btn_label = "▼ Collapse" if is_expanded else "▶ Expand for details"
            if st.button(btn_label, key=f"topic_{i}"):
                if is_expanded:
                    st.session_state.expanded_topics.remove(i)
                else:
                    st.session_state.expanded_topics.append(i)
                st.rerun()

            if is_expanded:
                detail_key = f"detail_{i}"
                if detail_key not in st.session_state:
                    with st.spinner(f"Loading details for '{t.get('title','')}' ..."):
                        try:
                            detail = get_topic_detail(
                                api_key,
                                t.get("title",""),
                                t.get("description",""),
                                d.get("summary","")
                            )
                            st.session_state[detail_key] = detail
                        except Exception as e:
                            st.session_state[detail_key] = f"Could not load: {e}"

                st.markdown(f"""
                <div class="card" style="border-color:{col}60; margin-top:-0.5rem;">
                  <div style="font-size:0.92rem; color:#ddd; line-height:1.8;">
                    {st.session_state.get(detail_key,'')}
                  </div>
                </div>""", unsafe_allow_html=True)

    # ── Mind Map ──────────────────────────────────────────────────────────────
    with tab3:
        mm = d.get("mindmap", {})
        if mm:
            svg = render_mindmap(mm)
            st.markdown(f'<div class="mindmap-container">{svg}</div>', unsafe_allow_html=True)
        else:
            st.info("No mind map data available.")

    # ── Flashcards ────────────────────────────────────────────────────────────
    with tab4:
        cards = d.get("flashcards", [])
        if cards:
            idx = st.session_state.fc_index % len(cards)
            card = cards[idx]
            flipped = st.session_state.fc_flipped

            st.markdown(f"**Card {idx+1} of {len(cards)}**")

            st.markdown(f"""
            <div class="flashcard">
              <div class="fc-question">❓ {card.get('question','')}</div>
              {'<div class="fc-answer">💡 ' + card.get("answer","") + '</div>' if flipped else '<div style="color:#555; font-size:0.85rem;">Click Flip to reveal answer</div>'}
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("⬅️ Prev"):
                    st.session_state.fc_index = (idx - 1) % len(cards)
                    st.session_state.fc_flipped = False
                    st.rerun()
            with c2:
                if st.button("🔄 Flip Card"):
                    st.session_state.fc_flipped = not flipped
                    st.rerun()
            with c3:
                if st.button("➡️ Next"):
                    st.session_state.fc_index = (idx + 1) % len(cards)
                    st.session_state.fc_flipped = False
                    st.rerun()
        else:
            st.info("No flashcards generated.")

else:
    st.markdown("""
    <div class="card" style="text-align:center; padding:3rem;">
      <div style="font-size:3rem; margin-bottom:1rem;">📂</div>
      <div class="card-title" style="font-size:1.2rem;">Upload a document to get started</div>
      <div class="card-desc">Enter your Gemini API key and upload a PDF or text file using the sidebar on the left</div>
    </div>
    """, unsafe_allow_html=True)
