"""
Visualization and display helpers for RAG exploration notebooks.
"""

from collections import Counter

import pandas as pd
from IPython.display import HTML, display
from langchain_core.documents import Document


def display_doc_preview(doc: Document, max_chars: int = 500) -> None:
    """Display a single document with metadata and content preview."""
    meta = doc.metadata
    content = doc.page_content[:max_chars]
    if len(doc.page_content) > max_chars:
        content += "..."

    source = meta.get('source', 'Unknown')
    file_type = meta.get('file_type', '?')
    page = meta.get('page', '?')

    html = f"""
    <div style="border:1px solid #ddd; border-radius:8px; padding:12px; margin:8px 0;
                background:#fafafa; font-family:monospace; font-size:13px;">
        <div style="margin-bottom:8px;">
            <b>{source}</b>
            <span style="color:#888; margin-left:12px;">[{file_type}]</span>
            <span style="color:#888; margin-left:12px;">page {page}</span>
        </div>
        <div style="white-space:pre-wrap; font-size:12px; color:#333; max-height:200px; overflow-y:auto;">
{content}
        </div>
        <div style="color:#888; font-size:11px; margin-top:6px;">
            {len(doc.page_content):,} characters
        </div>
    </div>
    """
    display(HTML(html))


def display_retrieval_results(
    query: str,
    results: list[tuple[Document, float]],
    max_content_chars: int = 300,
) -> None:
    """Display retrieval results with scores and content previews."""
    html = f'<h4 style="margin-bottom:4px;">Query: <i>"{query}"</i></h4>'

    for i, (doc, score) in enumerate(results, 1):
        content = doc.page_content[:max_content_chars]
        if len(doc.page_content) > max_content_chars:
            content += "..."

        score_color = "#2ecc71" if score > 0.7 else "#f39c12" if score > 0.4 else "#e74c3c"
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', '?')

        html += f"""
        <div style="border-left:4px solid {score_color}; padding:8px 12px; margin:6px 0;
                    background:#fafafa; font-size:12px;">
            <b>#{i}</b> — Score: <b style="color:{score_color}">{score:.4f}</b>
            — <i>{source}</i>
            <span style="color:#888;">[page {page}]</span>
            <div style="white-space:pre-wrap; color:#555; margin-top:4px; font-size:11px;">
{content}
            </div>
        </div>
        """

    display(HTML(html))


def corpus_summary_table(docs: list[Document]) -> pd.DataFrame:
    """Build a summary DataFrame of the corpus."""
    records = []
    for doc in docs:
        records.append({
            "source": doc.metadata.get("source", "Unknown"),
            "file_type": doc.metadata.get("file_type", "unknown"),
            "page": doc.metadata.get("page", "?"),
            "chars": len(doc.page_content),
        })
    return pd.DataFrame(records)


def section_breakdown(docs: list[Document]) -> pd.DataFrame:
    """Return a per-file-type summary."""
    df = corpus_summary_table(docs)
    return (
        df.groupby("file_type")
        .agg(
            count=("chars", "size"),
            total_chars=("chars", "sum"),
            avg_chars=("chars", "mean"),
            min_chars=("chars", "min"),
            max_chars=("chars", "max"),
        )
        .sort_values("count", ascending=False)
    )


def chunk_stats_table(chunks: list[Document]) -> pd.DataFrame:
    """Build a summary table of chunk sizes."""
    sizes = [len(c.page_content) for c in chunks]
    return pd.DataFrame({
        "metric": ["count", "mean", "std", "min", "25%", "50%", "75%", "max"],
        "chars": [
            len(sizes),
            pd.Series(sizes).mean(),
            pd.Series(sizes).std(),
            min(sizes),
            pd.Series(sizes).quantile(0.25),
            pd.Series(sizes).median(),
            pd.Series(sizes).quantile(0.75),
            max(sizes),
        ],
    }).set_index("metric").round(0).astype(int)


# ── Phase 4: Retrieval comparison display helpers ─────────────

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def display_strategy_comparison(df: pd.DataFrame) -> None:
    """Display a styled table highlighting best/worst per metric.

    Expects a DataFrame with strategies as rows and metrics as columns.
    Numeric columns are highlighted: green for best, red for worst.
    For latency_ms, lower is better; for all others, higher is better.
    """
    lower_better = {"latency_ms", "avg_latency_ms"}
    numeric_cols = df.select_dtypes(include="number").columns

    def highlight(col: pd.Series) -> list[str]:
        styles = [""] * len(col)
        if col.name not in numeric_cols:
            return styles
        if col.isna().all():
            return styles
        is_lower = col.name in lower_better
        best_idx = col.idxmin() if is_lower else col.idxmax()
        worst_idx = col.idxmax() if is_lower else col.idxmin()
        for i, idx in enumerate(col.index):
            if idx == best_idx:
                styles[i] = "background-color: #d4edda; font-weight: bold"
            elif idx == worst_idx:
                styles[i] = "background-color: #f8d7da"
        return styles

    styled = df.style.apply(highlight, axis=0).format(
        {c: "{:.4f}" for c in numeric_cols if c != "latency_ms" and c != "avg_latency_ms"},
    ).format(
        {c: "{:.1f}" for c in numeric_cols if c in lower_better},
    )
    display(HTML(styled.to_html()))


def display_category_breakdown(
    df: pd.DataFrame,
    metric: str = "precision_at_k",
    title: str = "Performance par categorie",
) -> None:
    """Display a grouped bar chart comparing strategies by query category.

    Args:
        df: DataFrame with columns: strategy, category, and the metric.
        metric: Which metric column to plot.
        title: Chart title.
    """
    pivot = df.pivot_table(
        index="category", columns="strategy", values=metric, aggfunc="mean",
    )

    ax = pivot.plot(kind="bar", figsize=(12, 5), width=0.8)
    ax.set_title(title, fontsize=14)
    ax.set_ylabel(metric)
    ax.set_xlabel("")
    ax.legend(title="Strategie", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def display_latency_comparison(df: pd.DataFrame) -> None:
    """Display a horizontal bar chart of average latency per strategy.

    Expects a DataFrame with 'strategy' and 'avg_latency_ms' columns.
    """
    df_sorted = df.sort_values("avg_latency_ms")
    colors = sns.color_palette("viridis", len(df_sorted))

    fig, ax = plt.subplots(figsize=(10, max(4, len(df_sorted) * 0.5)))
    ax.barh(df_sorted["strategy"], df_sorted["avg_latency_ms"], color=colors)
    ax.set_xlabel("Latence moyenne (ms)")
    ax.set_title("Latence par strategie de retrieval")

    for i, (_, row) in enumerate(df_sorted.iterrows()):
        ax.text(
            row["avg_latency_ms"] + 0.5, i,
            f'{row["avg_latency_ms"]:.1f} ms',
            va="center", fontsize=10,
        )

    plt.tight_layout()
    plt.show()


def display_radar_chart(
    df: pd.DataFrame,
    metrics: list[str],
    strategy_col: str = "strategy",
    title: str = "Comparaison radar des strategies",
) -> None:
    """Display a radar chart comparing strategies across multiple metrics.

    Args:
        df: DataFrame with one row per strategy.
        metrics: List of metric column names to plot.
        strategy_col: Column containing strategy names.
        title: Chart title.
    """
    strategies = df[strategy_col].tolist()
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    colors = sns.color_palette("husl", len(strategies))

    for i, strategy in enumerate(strategies):
        row = df[df[strategy_col] == strategy].iloc[0]
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=strategy, color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()
