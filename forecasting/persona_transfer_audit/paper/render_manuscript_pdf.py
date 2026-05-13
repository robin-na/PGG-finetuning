"""Render the persona-transfer LaTeX manuscript to a readable PDF.

The local environment for this project does not currently include a TeX engine.
This script keeps the LaTeX source as the editable manuscript while producing a
review PDF with ReportLab from the same source.
"""

from __future__ import annotations

import argparse
import html
import hashlib
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

CITE_LABELS = {
    "aher2023turing": "Aher et al. 2023",
    "akata2025repeatedgames": "Akata et al. 2025",
    "argyle2023silicon": "Argyle et al. 2023",
    "gao2024survey": "Gao et al. 2024",
    "gui2025causal": "Gui and Toubia 2025",
    "hu2025simbench": "Hu et al. 2025",
    "li2025catch": "Li et al. 2025",
    "lutz2025prompt": "Lutz et al. 2025",
    "paglieri2026persona": "Paglieri et al. 2026",
    "park2023generative": "Park et al. 2023",
    "vezhnevets2023concordia": "Vezhnevets et al. 2023",
}


def _format_math(expr: str) -> str:
    def plain(markup: str) -> str:
        markup = re.sub(r"<sub>(.*?)</sub>", r"_\1", markup)
        markup = re.sub(r"<sup>(.*?)</sup>", r"^\1", markup)
        return markup

    expr = re.sub(r"\s+", " ", expr.strip())
    exact = {
        r"p \in \mathcal{P}": "p ∈ P",
        r"g \in \mathcal{G}": "g ∈ G",
        r"C_g=\{1,\ldots,J_g\}": "C<sub>g</sub> = {1, ..., J<sub>g</sub>}",
        r"T_{gj}": "T<sub>gj</sub>",
        r"r=(p,g)": "r = (p, g)",
        r"q_r": "q<sub>r</sub>",
        r"C_g": "C<sub>g</sub>",
        r"q_{rj}\geq 0": "q<sub>rj</sub> ≥ 0",
        r"\sum_{j\in C_g} q_{rj}=1": "∑<sub>j∈C<sub>g</sub></sub> q<sub>rj</sub> = 1",
        r"m_r=\arg\max_{j\in C_g} q_{rj}": "m<sub>r</sub> = arg max<sub>j∈C<sub>g</sub></sub> q<sub>rj</sub>",
        r"F:(p,T_g)\mapsto q_r": "F: (p, T<sub>g</sub>) → q<sub>r</sub>",
        r"n_{gj}=\sum_{r:g(r)=g}\mathbf{1}\{m_r=j\}": "n<sub>gj</sub> = ∑<sub>r:g(r)=g</sub> 1{m<sub>r</sub>=j}",
        r"L_g=\max_{j\in C_g} n_{gj}/N_g": "L<sub>g</sub> = max<sub>j∈C<sub>g</sub></sub> n<sub>gj</sub>/N<sub>g</sub>",
        r"N_g=\sum_j n_{gj}": "N<sub>g</sub> = ∑<sub>j</sub> n<sub>gj</sub>",
        r"m_r \sim \mathrm{Uniform}(C_g)": "m<sub>r</sub> ∼ Uniform(C<sub>g</sub>)",
        r"L_g": "L<sub>g</sub>",
        r"(n_{g1},\ldots,n_{gJ_g})": "(n<sub>g1</sub>, ..., n<sub>gJ<sub>g</sub></sub>)",
        r"I": "I",
        r"|\{i\in I:n_i>0\}|/|I|": "|{i∈I : n<sub>i</sub> > 0}| / |I|",
        r"\exp[-\sum_i s_i\log(s_i)]/|I|": "exp[-∑<sub>i</sub> s<sub>i</sub> log(s<sub>i</sub>)] / |I|",
        r"s_i=n_i/\sum_i n_i": "s<sub>i</sub> = n<sub>i</sub> / ∑<sub>i</sub> n<sub>i</sub>",
        r"\sum_{i\in \mathrm{Top}_{0.05}(I)} s_i": "∑<sub>i∈Top<sub>0.05</sub>(I)</sub> s<sub>i</sub>",
        r"m_r": "m<sub>r</sub>",
        r"C_{g(r)}": "C<sub>g(r)</sub>",
        r"Y_{gj}": "Y<sub>gj</sub>",
        r"\sum_{j\in C_g}q_{rj}Y_{gj}": "∑<sub>j∈C<sub>g</sub></sub> q<sub>rj</sub>Y<sub>gj</sub>",
        r"J_g^{-1}\sum_{j\in C_g}Y_{gj}": "J<sub>g</sub><sup>-1</sup>∑<sub>j∈C<sub>g</sub></sub>Y<sub>gj</sub>",
        r"\Delta_Y=R^{-1}\sum_r[\sum_{j\in C_{g(r)}}q_{rj}Y_{g(r)j}-J_{g(r)}^{-1}\sum_{j\in C_{g(r)}}Y_{g(r)j}]": "Δ<sub>Y</sub> = R<sup>-1</sup>∑<sub>r</sub>[∑<sub>j∈C<sub>g(r)</sub></sub>q<sub>rj</sub>Y<sub>g(r)j</sub> - J<sub>g(r)</sub><sup>-1</sup>∑<sub>j∈C<sub>g(r)</sub></sub>Y<sub>g(r)j</sub>]",
        r"\Delta_Y": "Δ<sub>Y</sub>",
        r"Y": "Y",
        r"N": "N",
    }
    if expr in exact:
        return plain(exact[expr])
    expr = expr.replace(r"\mathcal{P}", "P").replace(r"\mathcal{G}", "G")
    expr = expr.replace(r"\ldots", "...").replace(r"\geq", "≥").replace(r"\leq", "≤")
    expr = expr.replace(r"\sum", "∑").replace(r"\exp", "exp").replace(r"\log", "log")
    expr = expr.replace(r"\in", "∈").replace(r"\sim", "∼").replace(r"\mapsto", "→")
    expr = expr.replace(r"\mathrm{Uniform}", "Uniform").replace(r"\mathrm{Top}", "Top")
    expr = expr.replace(r"\mathbf{1}", "1")
    expr = expr.replace(r"\{", "{").replace(r"\}", "}").replace("\\", "")
    expr = re.sub(r"_\{([^{}]+)\}", r"_\1", expr)
    expr = re.sub(r"\^\{([^{}]+)\}", r"^\1", expr)
    return expr


def _format_cites(keys: str) -> str:
    return "; ".join(CITE_LABELS.get(key.strip(), key.strip()) for key in keys.split(","))


def _clean_inline(text: str) -> str:
    text = text.strip()
    text = re.sub(r"%.*$", "", text)
    text = text.replace("``", '"').replace("''", '"')
    text = text.replace("~", " ")
    text = re.sub(r"\\\((.*?)\\\)", lambda match: _format_math(match.group(1)), text)
    text = re.sub(r"\\%", "%", text)
    text = re.sub(r"\\_", "_", text)
    text = re.sub(r"\\&", "&", text)
    text = re.sub(r"\\mathcal\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\mathbf\{([^{}]+)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", text)
    replacements = {
        r"\ldots": "...",
        r"\geq": ">=",
        r"\leq": "<=",
        r"\neq": "!=",
        r"\in": " in ",
        r"\sim": "~",
        r"\sum": "sum",
        r"\max": "max",
        r"\arg": "arg",
        r"\exp": "exp",
        r"\log": "log",
        r"\Delta": "Delta",
        r"\mapsto": "->",
        r"\times": "x",
        r"\left": "",
        r"\right": "",
        r"\{": "{",
        r"\}": "}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\texttt\{([^{}]+)\}", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\\textit\{([^{}]+)\}", r"<i>\1</i>", text)
    text = re.sub(r"\\textbf\{([^{}]+)\}", r"<b>\1</b>", text)
    text = re.sub(r"\\citet\{([^{}]+)\}", lambda match: _format_cites(match.group(1)), text)
    text = re.sub(r"\\citep\{([^{}]+)\}", lambda match: f"({_format_cites(match.group(1))})", text)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)
    text = text.replace(r"\(", "").replace(r"\)", "")
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\\", "")
    text = text.replace("$", "")
    return text


def _paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    # Preserve a tiny subset of markup inserted by _clean_inline.
    safe = html.escape(text)
    safe = safe.replace("&lt;i&gt;", "<i>").replace("&lt;/i&gt;", "</i>")
    safe = safe.replace("&lt;b&gt;", "<b>").replace("&lt;/b&gt;", "</b>")
    safe = safe.replace("&lt;sub&gt;", "<sub>").replace("&lt;/sub&gt;", "</sub>")
    safe = safe.replace("&lt;sup&gt;", "<sup>").replace("&lt;/sup&gt;", "</sup>")
    safe = safe.replace("&lt;font name=&#x27;Courier&#x27;&gt;", "<font name='Courier'>")
    safe = safe.replace("&lt;/font&gt;", "</font>")
    return Paragraph(safe, style)


def _parse_bib_entries(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    source = path.read_text(encoding="utf-8")
    entries: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)(?=\n@\w+\s*\{|\Z)", source, re.S):
        key = match.group(1).strip()
        body = match.group(2)
        fields: dict[str, str] = {}
        for field in re.finditer(r"(\w+)\s*=\s*\{(.*?)\}\s*,?", body, re.S):
            name = field.group(1).lower()
            value = re.sub(r"\s+", " ", field.group(2).strip())
            value = value.replace("{", "").replace("}", "")
            value = _clean_inline(value)
            fields[name] = value
        entries[key] = fields
    return entries


def _format_bib_entry(fields: dict[str, str]) -> str:
    author = fields.get("author", "").replace(" and ", ", ")
    year = fields.get("year", "")
    title = fields.get("title", "")
    if "journal" in fields:
        venue = fields["journal"]
        if fields.get("volume"):
            venue += f" {fields['volume']}"
            if fields.get("number"):
                venue += f"({fields['number']})"
        if fields.get("pages"):
            venue += f": {fields['pages']}"
    elif "booktitle" in fields:
        venue = fields["booktitle"]
        if fields.get("series"):
            venue += f", {fields['series']}"
        if fields.get("pages"):
            venue += f", {fields['pages']}"
    elif fields.get("archiveprefix") == "arXiv" or fields.get("eprint"):
        venue = f"arXiv:{fields.get('eprint', '')}"
    else:
        venue = fields.get("publisher", "")
    doi = f" doi:{fields['doi']}." if fields.get("doi") else ""
    url = f" {fields['url']}." if fields.get("url") else ""
    return f"{author} ({year}). {title}. {venue}.{doi}{url}".replace("--", "-").strip()


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=18,
            leading=22,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "author": ParagraphStyle(
            "Author",
            parent=base["Normal"],
            fontName="Times-Roman",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            spaceAfter=18,
        ),
        "section": ParagraphStyle(
            "Section",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=13,
            leading=16,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "subsection": ParagraphStyle(
            "Subsection",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=11.5,
            leading=14,
            spaceBefore=10,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=10.2,
            leading=13.2,
            firstLineIndent=14,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "abstract": ParagraphStyle(
            "Abstract",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9.7,
            leading=12.4,
            alignment=TA_JUSTIFY,
            leftIndent=18,
            rightIndent=18,
            spaceAfter=6,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=8.8,
            leading=11,
            alignment=TA_JUSTIFY,
            spaceBefore=4,
            spaceAfter=10,
        ),
        "refs": ParagraphStyle(
            "References",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9,
            leading=11,
            leftIndent=14,
            firstLineIndent=-14,
            spaceAfter=4,
        ),
    }


def _page_footer(canvas, doc) -> None:  # noqa: ANN001
    canvas.saveState()
    canvas.setFont("Times-Roman", 9)
    canvas.setFillColor(colors.HexColor("#555555"))
    canvas.drawCentredString(letter[0] / 2, 0.45 * inch, str(doc.page))
    canvas.restoreState()


def _render_equation(math: str, output_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception:
        return None

    math = math.strip().rstrip(".")
    math = math.replace(r"\ge ", r"\geq ").replace(r"\le ", r"\leq ")
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(("v2:" + math).encode("utf-8")).hexdigest()[:12]
    output_path = output_dir / f"equation_{digest}.png"
    if output_path.exists():
        return output_path

    fig = plt.figure(figsize=(6.4, 0.42))
    fig.text(0.5, 0.5, f"${math}$", ha="center", va="center", fontsize=9)
    fig.savefig(output_path, bbox_inches="tight", dpi=300, transparent=True, pad_inches=0.03)
    plt.close(fig)
    return output_path


def parse_latex(path: Path) -> tuple[str, str, list[tuple[str, str | tuple[Path, str]]]]:
    source = path.read_text(encoding="utf-8")
    title = re.search(r"\\title\{(.+?)\}", source, re.S)
    author = re.search(r"\\author\{(.+?)\}", source, re.S)
    title_text = _clean_inline(title.group(1)) if title else "Manuscript"
    author_text = _clean_inline(author.group(1)) if author else ""

    body = source.split(r"\begin{document}", 1)[-1].split(r"\end{document}", 1)[0]
    body = re.sub(r"\\maketitle", "", body)
    cited_keys: list[str] = []
    seen_cites: set[str] = set()
    for cite_match in re.finditer(r"\\cite[tp]?\{([^{}]+)\}", body):
        for key in cite_match.group(1).split(","):
            key = key.strip()
            if key and key not in seen_cites:
                cited_keys.append(key)
                seen_cites.add(key)

    items: list[tuple[str, str | tuple[Path, str]]] = []
    paragraph: list[str] = []
    in_abstract = False
    in_refs = False
    in_figure = False
    in_equation = False
    equation_lines: list[str] = []
    figure_path: Path | None = None
    figure_caption = ""

    def flush() -> None:
        nonlocal paragraph
        text = " ".join(part.strip() for part in paragraph if part.strip())
        paragraph = []
        if text:
            items.append(("abstract" if in_abstract else "refs" if in_refs else "body", _clean_inline(text)))

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if in_equation:
            if line.startswith(r"\]"):
                equation = " ".join(equation_lines).strip()
                if equation:
                    items.append(("equation", equation))
                equation_lines = []
                in_equation = False
            else:
                equation_lines.append(line)
            continue
        if not line:
            flush()
            continue
        if line.startswith(r"\["):
            flush()
            in_equation = True
            equation_lines = []
            remainder = line[2:].strip()
            if remainder:
                equation_lines.append(remainder)
            continue
        if line.startswith(r"\begin{abstract}"):
            flush()
            in_abstract = True
            items.append(("subsection", "Abstract"))
            continue
        if line.startswith(r"\end{abstract}"):
            flush()
            in_abstract = False
            continue
        if line.startswith(r"\section"):
            flush()
            items.append(("section", _clean_inline(re.search(r"\{(.+)\}", line).group(1))))
            continue
        if line.startswith(r"\subsection"):
            flush()
            items.append(("subsection", _clean_inline(re.search(r"\{(.+)\}", line).group(1))))
            continue
        if line.startswith(r"\begin{figure}"):
            flush()
            in_figure = True
            figure_path = None
            figure_caption = ""
            continue
        if in_figure and line.startswith(r"\includegraphics"):
            match = re.search(r"\{(.+?)\}", line)
            if match:
                figure_path = (path.parent / match.group(1)).resolve()
            continue
        if in_figure and line.startswith(r"\caption"):
            match = re.search(r"\{(.+)\}", line)
            figure_caption = _clean_inline(match.group(1)) if match else ""
            continue
        if line.startswith(r"\end{figure}"):
            if figure_path:
                items.append(("figure", (figure_path, figure_caption)))
            in_figure = False
            continue
        if in_figure:
            continue
        if line.startswith(r"\begin{thebibliography}"):
            flush()
            in_refs = True
            items.append(("section", "References"))
            continue
        if line.startswith(r"\end{thebibliography}"):
            flush()
            in_refs = False
            continue
        if line.startswith(r"\bibliographystyle"):
            flush()
            continue
        if line.startswith(r"\bibliography"):
            flush()
            match = re.search(r"\{(.+)\}", line)
            bib_names = [name.strip() for name in match.group(1).split(",")] if match else []
            bib_entries: dict[str, dict[str, str]] = {}
            for name in bib_names:
                bib_path = (path.parent / f"{name}.bib").resolve()
                bib_entries.update(_parse_bib_entries(bib_path))
            items.append(("section", "References"))
            keys = cited_keys or list(bib_entries)
            for key in keys:
                if key in bib_entries:
                    items.append(("refs", _format_bib_entry(bib_entries[key])))
            continue
        if line.startswith(r"\bibitem"):
            flush()
            line = re.sub(r"\\bibitem(?:\[[^\]]+\])?\{[^{}]+\}", "", line).strip()
            if line:
                paragraph.append(line)
            continue
        if line.startswith("\\") and not line.startswith("\\[") and not line.startswith("\\]"):
            continue
        paragraph.append(line)
    flush()
    return title_text, author_text, items


def render(tex_path: Path, output_path: Path) -> None:
    styles = _styles()
    title, author, items = parse_latex(tex_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=0.85 * inch,
        rightMargin=0.85 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.7 * inch,
    )
    story = [_paragraph(title, styles["title"])]
    if author:
        story.append(_paragraph(author, styles["author"]))
    for kind, value in items:
        if kind == "section":
            story.append(_paragraph(str(value), styles["section"]))
        elif kind == "subsection":
            story.append(_paragraph(str(value), styles["subsection"]))
        elif kind == "figure":
            image_path, caption = value  # type: ignore[misc]
            if image_path.exists():
                image = Image(str(image_path))
                max_width = 6.75 * inch
                scale = min(max_width / image.imageWidth, 1.0)
                image.drawWidth = image.imageWidth * scale
                image.drawHeight = image.imageHeight * scale
                story.extend([Spacer(1, 8), image, _paragraph(caption, styles["caption"])])
        elif kind == "equation":
            equation_path = _render_equation(str(value), output_path.parent / "_rendered_equations")
            if equation_path and equation_path.exists():
                image = Image(str(equation_path))
                max_width = 6.0 * inch
                max_height = 0.5 * inch
                scale = min(max_width / image.imageWidth, max_height / image.imageHeight, 1.0)
                image.drawWidth = image.imageWidth * scale
                image.drawHeight = image.imageHeight * scale
                story.extend([Spacer(1, 4), image, Spacer(1, 4)])
            else:
                story.append(_paragraph(_clean_inline(str(value)), styles["body"]))
        else:
            story.append(_paragraph(str(value), styles[kind]))
    doc.build(story, onFirstPage=_page_footer, onLaterPages=_page_footer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, default=Path("manuscript.tex"))
    parser.add_argument("--output", type=Path, default=Path("manuscript.pdf"))
    args = parser.parse_args()
    render(args.tex.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
