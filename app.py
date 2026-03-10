"""Gradio demo UI for the Course Notes QA Assistant."""

import gradio as gr
from huggingface_hub import InferenceClient

from src.pdf_processor import process_pdf
from src.vectorizer import DocumentVectorizer
from src.search import search


def _upload_pdf(pdf_file) -> tuple[str, DocumentVectorizer | None]:
    """Process the uploaded PDF and index its content.

    Called when the user clicks "Process PDF".
    """
    if pdf_file is None:
        return "⚠️  Please upload a PDF file first.", None

    try:
        chunks = process_pdf(pdf_file)
        vectorizer = DocumentVectorizer()
        vectorizer.fit(chunks)
        return (
            f"✅  PDF processed — {len(chunks)} text chunks indexed and ready.",
            vectorizer
        )
    except Exception as exc:  # pylint: disable=broad-except
        return f"❌  Error processing PDF: {exc}", None


def _answer_question(
    question: str, history: list, vectorizer: DocumentVectorizer | None, api_key: str
) -> tuple[list, str]:
    """Retrieve the most relevant passages for *question* and synthesize an answer.

    Called when the user submits a question.
    """
    if not question.strip():
        return history, ""

    if not api_key.strip():
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "⚠️  Please enter your Hugging Face API Token first to generate answers."}
        ]
        return history, ""

    if vectorizer is None:
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": "⚠️  Please upload and process a PDF first."}
        ]
        return history, ""

    results = search(question, vectorizer, top_k=5)

    if not results or results[0][1] < 0.01:
        answer = (
            "I couldn't find relevant information in the document for your question."
        )
    else:
        context_parts = []
        for idx, (chunk, score) in enumerate(results, start=1):
            context_parts.append(f"--- Passage {idx} ---\n{chunk}")
        context_text = "\n\n".join(context_parts)

        prompt = (
            f"You are a helpful teaching assistant. Answer the student's question based ONLY on the provided course notes passages.\n\n"
            f"Context passages:\n{context_text}\n"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ]

        try:
            client = InferenceClient(api_key=api_key.strip())
            # Use Qwen 2.5 72B Instruct - powerful free model
            response = client.chat_completion(
                model="Qwen/Qwen2.5-72B-Instruct",
                messages=messages,
                max_tokens=600,
                temperature=0.3,
                top_p=0.9,
            )
            answer = response.choices[0].message.content
            
        except Exception as e:
            answer = f"⚠️  An error occurred generating the answer with Hugging Face API: {e}"

    history = history + [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ]
    return history, ""


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="Course Notes QA Assistant", theme=gr.themes.Soft()) as demo:
    vectorizer_state = gr.State()

    gr.Markdown(
        "# 📚 Course Notes QA Assistant\n"
        "Upload your PDF course notes, then ask questions to retrieve the most "
        "relevant passages. Uses Hugging Face free API for remote inference."
    )

    with gr.Row():
        # ── Left column: PDF upload ──────────────────────────────────────
        with gr.Column(scale=1):
            api_key_input = gr.Textbox(
                label="Hugging Face API Token (hf_...)",
                placeholder="hf_abc123...",
                type="password",
            )
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
    process_btn.click(_upload_pdf, inputs=[pdf_input], outputs=[status_box, vectorizer_state])

    ask_btn.click(
        _answer_question,
        inputs=[question_input, chatbot, vectorizer_state, api_key_input],
        outputs=[chatbot, question_input],
    )
    question_input.submit(
        _answer_question,
        inputs=[question_input, chatbot, vectorizer_state, api_key_input],
        outputs=[chatbot, question_input],
    )
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, question_input])


if __name__ == "__main__":
    demo.launch()
