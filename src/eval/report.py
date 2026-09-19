"""Small helper to keep reports/results.md as a single running document:
each phase's evaluation script writes/updates its own "## <title>" section
without clobbering sections written by other phases."""

from pathlib import Path


def format_metrics_table(rows: list, ks: list) -> str:
    """rows: list of (model_name, split_name, results_dict) where
    results_dict has 'recall@{k}'/'ndcg@{k}' for each k in ks, plus
    'coverage@{max(ks)}' and 'n_users_evaluated'."""
    max_k = max(ks)
    cols = (
        ["Model", "Split", "n_users"]
        + [f"Recall@{k}" for k in ks]
        + [f"NDCG@{k}" for k in ks]
        + [f"Coverage@{max_k}"]
    )
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for name, split_name, res in rows:
        cells = [name, split_name, str(res["n_users_evaluated"])]
        cells += [f"{res[f'recall@{k}']:.4f}" for k in ks]
        cells += [f"{res[f'ndcg@{k}']:.4f}" for k in ks]
        cells += [f"{res[f'coverage@{max_k}']:.4f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def update_section(path: Path, title: str, body_md: str) -> None:
    path = Path(path)
    heading = f"## {title}"
    section = f"{heading}\n\n{body_md.strip()}\n"

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# Results\n\n{section}\n")
        return

    text = path.read_text()
    lines = text.split("\n")
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        if line.strip() == heading:
            start = i
        elif start is not None and line.startswith("## "):
            end = i
            break

    if start is None:
        # Section doesn't exist yet: append it.
        new_text = text.rstrip("\n") + "\n\n" + section
    else:
        new_lines = lines[:start] + section.split("\n") + lines[end:]
        new_text = "\n".join(new_lines)
        # Collapse any accidental blank-line runs left by the splice.
        while "\n\n\n" in new_text:
            new_text = new_text.replace("\n\n\n", "\n\n")

    path.write_text(new_text)
