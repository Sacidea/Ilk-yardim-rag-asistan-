import os
from dotenv import load_dotenv
import anthropic
import chromadb
import gradio as gr
from sentence_transformers import SentenceTransformer
from flashrank import Ranker, RerankRequest

load_dotenv()

# ── 1. VERİ KATMANI ──
def load_documents(folder):
    import glob
    import json
    chunks = []

    for filepath in glob.glob(f"{folder}/*.txt"):
        print(f"  Yükleniyor: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        for block in content.split("\n\n"):
            block = block.strip()
            if block:
                chunks.append(block)

    for filepath in glob.glob(f"{folder}/*.json"):
        print(f"  Yükleniyor: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for intent in data.get("intents", []):
            tag = intent.get("tag", "")
            patterns = intent.get("patterns", [])
            responses = intent.get("responses", [])
            if responses and responses[0].strip():
                chunk = f"# {tag}\n"
                chunk += f"Sorular: {', '.join(patterns)}\n"
                chunk += f"Cevap: {responses[0]}"
                chunks.append(chunk)

    for filepath in glob.glob(f"{folder}/*.pdf"):
        import fitz
        print(f"  Yükleniyor: {filepath}")
        doc = fitz.open(filepath)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        for block in text.split("\n\n"):
            block = block.strip()
            if len(block) > 50:
                chunks.append(block)

    print(f"Toplam {len(chunks)} chunk yüklendi.")
    return chunks


# ── 2. EMBEDDING MODELİ ──
print("Model yükleniyor...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# ── 3. VECTOR DATABASE ──
client = chromadb.PersistentClient(path="chroma_db")
try:
    collection = client.get_collection("ilk_yardim")
    print("Mevcut index yüklendi!")
except:
    collection = client.create_collection("ilk_yardim")

def build_index(chunks):
    if collection.count() > 0:
        print(f"Index zaten mevcut ({collection.count()} chunk), atlanıyor.")
        return
    print(f"{len(chunks)} chunk indexleniyor...")
    embeddings = model.encode(chunks).tolist()
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    print("Vector DB hazır!")


# ── 4. RETRIEVER + RERANKER ──
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")

def retrieve(query, top_k=2):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=6
    )
    candidates = [{"text": doc} for doc in results["documents"][0]]
    rerank_request = RerankRequest(query=query, passages=candidates)
    reranked = ranker.rerank(rerank_request)
    top_chunks = [r["text"] for r in reranked[:top_k]]
    return top_chunks


# ── 5. LLM ──
llm = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def stream_answer(message, history):
    chunks = retrieve(message)
    context = "\n\n---\n\n".join(chunks)

    panel = "### 📎 Retrieve Edilen Kaynaklar\n\n"
    for i, chunk in enumerate(chunks):
        title = chunk.split("\n")[0].replace("#", "").strip()
        preview = chunk.split("\n")[1][:80] if len(chunk.split("\n")) > 1 else ""
        panel += f"**{i+1}. {title}**\n_{preview}..._\n\n---\n"

    system = f"""Sen bir Türkçe ilk yardım asistanısın.
Yalnızca aşağıdaki bilgi tabanını kullan. Bilgi tabanında yoksa 'Bilgi tabanımda bu konu yok' de.
Acil durumlarda mutlaka 112'yi hatırlat.

BİLGİ TABANI:
{context}"""

    messages = []
    for h in history:
        if isinstance(h, dict):
            messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": message})

    full_answer = ""
    with llm.messages.stream(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system,
        messages=messages
    ) as stream:
        for text in stream.text_stream:
            full_answer += text
            yield full_answer, panel

    yield full_answer, panel


# ── 6. DOSYA YÜKLEME ──
def upload_files(files):
    if not files:
        return "⚠️ Dosya seçilmedi."
    import shutil
    import fitz
    import json as json_lib

    count = 0
    for file in files:
        filename = os.path.basename(file.name)
        dest = os.path.join("data", filename)
        shutil.copy(file.name, dest)

        new_chunks = []
        if filename.endswith(".txt"):
            with open(dest, "r", encoding="utf-8") as f:
                content = f.read()
            for block in content.split("\n\n"):
                block = block.strip()
                if block:
                    new_chunks.append(block)

        elif filename.endswith(".json"):
            with open(dest, "r", encoding="utf-8") as f:
                data = json_lib.load(f)
            for intent in data.get("intents", []):
                tag = intent.get("tag", "")
                patterns = intent.get("patterns", [])
                responses = intent.get("responses", [])
                if responses and responses[0].strip():
                    chunk = f"# {tag}\nSorular: {', '.join(patterns)}\nCevap: {responses[0]}"
                    new_chunks.append(chunk)

        elif filename.endswith(".pdf"):
            doc = fitz.open(dest)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            for block in text.split("\n\n"):
                block = block.strip()
                if len(block) > 50:
                    new_chunks.append(block)

        if new_chunks:
            embeddings = model.encode(new_chunks).tolist()
            start_id = collection.count()
            collection.add(
                documents=new_chunks,
                embeddings=embeddings,
                ids=[f"chunk_{start_id + i}" for i in range(len(new_chunks))]
            )
            count += len(new_chunks)

    return f"✅ {len(files)} dosya yüklendi, {count} yeni chunk indexlendi!"


# ── 7. GRADIO ARAYÜZÜ ──
print("Veri yükleniyor...")
all_chunks = load_documents("data")
build_index(all_chunks)

with gr.Blocks(title="İlk Yardım RAG Chatbot") as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <h1 style="font-size:2rem; margin:0;">🚑 İlk Yardım Asistanı</h1>
        <p style="color:gray; margin-top:6px;">RAG destekli · Bilgi tabanından gerçek zamanlı retrieval</p>
        <div style="display:inline-block; background:#fee2e2; color:#991b1b;
                    padding:6px 16px; border-radius:20px; font-size:13px; margin-top:8px;">
            ⚠️ Bu uygulama tıbbi tavsiye vermez. Acil durumlarda 112'yi arayın.
        </div>
    </div>
    """)

    history_state = gr.State([])

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=480, label="Sohbet")
            msg = gr.Textbox(
                placeholder="Sorunuzu yazın... (örn: CPR nasıl yapılır?)",
                label="",
                lines=1
            )
            with gr.Row():
                send = gr.Button("Gönder 🔍", variant="primary")
                clear = gr.Button("Temizle")

            gr.Examples(
                examples=[
                    "Kalp krizi belirtileri neler?",
                    "CPR nasıl yapılır?",
                    "Yanığa ne iyi gelir?",
                    "Anafilaktik şok nedir?",
                    "Burun kanaması nasıl durdurulur?",
                ],
                inputs=msg
            )

            with gr.Accordion("📄 Bilgi Tabanına Dosya Ekle", open=False):
                file_upload = gr.File(
                    label="TXT, JSON veya PDF yükle",
                    file_types=[".txt", ".json", ".pdf"],
                    file_count="multiple"
                )
                upload_btn = gr.Button("Yükle ve İndeksle", variant="secondary")
                upload_status = gr.Markdown("")

        with gr.Column(scale=1):
            chunk_panel = gr.Markdown(
                value="### 📎 Retrieve Edilen Kaynaklar\n\nBir soru sorun, ilgili kaynaklar burada görünecek.",
            )

    def respond(message, history):
        if not message.strip():
            yield history, "### 📎 Retrieve Edilen Kaynaklar\n\nBir soru sorun.", history, ""
            return
        history = history or []
        answer = ""
        panel = ""
        for answer, panel in stream_answer(message, history):
            display = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": answer}
            ]
            yield display, panel, history, ""
        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": answer}
        ]
        yield history, panel, history, ""

    send.click(
        respond,
        inputs=[msg, history_state],
        outputs=[chatbot, chunk_panel, history_state, msg]
    )
    msg.submit(
        respond,
        inputs=[msg, history_state],
        outputs=[chatbot, chunk_panel, history_state, msg]
    )
    clear.click(
        lambda: ([], [], "### 📎 Retrieve Edilen Kaynaklar\n\nBir soru sorun."),
        outputs=[chatbot, history_state, chunk_panel]
    )
    upload_btn.click(
        upload_files,
        inputs=[file_upload],
        outputs=[upload_status]
    )

demo.launch()