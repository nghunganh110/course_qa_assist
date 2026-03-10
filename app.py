"""Gradio demo UI for the Course Notes QA Assistant."""

import gradio as gr

from src.pdf_processor import process_pdf
from src.vectorizer import DocumentVectorizer
from src.search import search

# ---------------------------------------------------------------------------
# Session-level state holders (replaced on each new PDF upload)
# ---------------------------------------------------------------------------
_vectorizer: DocumentVectorizer | None = None


def _upload_pdf(pdf_file) -> str:
    """Process the uploaded PDF and index its content.

    Called when the user clicks "Process PDF".
    """
    global _vectorizer

    if pdf_file is None:
        return "⚠️  Please upload a PDF file first."

    try:
        chunks = process_pdf(pdf_file)
        _vectorizer = DocumentVectorizer()
        _vectorizer.fit(chunks)
        return (
            f"✅  PDF processed — {len(chunks)} text chunks indexed and ready."
        )
    except Exception as exc:  # pylint: disable=broad-except
        _vectorizer = None
        return f"❌  Error processing PDF: {exc}"


def _answer_question(
    question: str, history: list
) -> tuple[list, str]:
    """Retrieve the most relevant passages for *question*.

    Called when the user submits a question.
    """
    if not question.strip():
        return history, ""

    if _vectorizer is None:
        history = history + [[question, "⚠️  Please upload and process a PDF first."]]
        return history, ""

    results = search(question, _vectorizer, top_k=3)

    if not results or results[0][1] < 0.01:
        answer = (
            "I couldn't find relevant information in the document for your question."
        )
    else:
        parts = []
        for idx, (chunk, score) in enumerate(results, start=1):
            parts.append(f"**Passage {idx}** *(relevance: {score:.3f})*\n\n{chunk}")
        answer = "\n\n---\n\n".join(parts)

    history = history + [[question, answer]]
    return history, ""


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Course Notes QA Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 📚 Course Notes QA Assistant\n"
        "Upload your PDF course notes, then ask questions to retrieve the most "
        "relevant passages."
    )

    with gr.Row():
        # ── Left column: PDF upload ──────────────────────────────────────
        with gr.Column(scale=1):
            pdf_input = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath",
            )
            process_btn = gr.Button("Process PDF", variant="primary")
            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2,
            )

        # ── Right column: chat interface ─────────────────────────────────
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Retrieved Passages", height=450)
            with gr.Row():
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask a question about your course notes…",
                    scale=4,
                )
                ask_btn = gr.Button("Ask", variant="primary", scale=1)
            clear_btn = gr.Button("Clear Chat")

    # ── Event wiring ─────────────────────────────────────────────────────
    process_btn.click(_upload_pdf, inputs=[pdf_input], outputs=[status_box])

    ask_btn.click(
        _answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input],
    )
    question_input.submit(
        _answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot, question_input],
    )
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, question_input])


if __name__ == "__main__":
    demo.launch()
