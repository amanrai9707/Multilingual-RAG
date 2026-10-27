"""Convert a source PDF into the clean, page-marked text file the pipeline needs.

    python -m src.pdf_to_clean path/to/report.pdf --doc-id WII-001 --out data/clean [--ocr] [--layout]

Output:  data/clean/<doc_id>.txt          text with  <<<PAGE n>>>  markers (n = the page number your PDF viewer shows)
         data/clean/<doc_id>.report.json  what was removed / which pages need a human look

PAGE NUMBERS: by default n is the PDF viewer's page number (1 = first page of the file), which is unambiguous
to cite. If you would rather cite the number PRINTED on the page, pass --page-offset (e.g. -4 when the printed
page 1 is the 5th PDF page). Pick one convention for the whole project and write it in Decisions_Log.

What it does:  extracts text page by page (poppler pdftotext) -> removes running headers/footers and bare page
numbers -> re-joins wrapped lines into paragraphs -> flags pages with no text layer (scanned/figure) and pages
that look like tables -> optionally OCRs the no-text pages with tesseract.

What it can NOT do: understand tables or charts. Numbers inside tables survive as text but lose their columns;
a person must transcribe any table value that a question depends on (put it in the question's supporting_passage).
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile
from collections import Counter

PAGE_WORD_RE = re.compile(r"^\s*page\s+\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\s*$", re.I)      # "Page 7", "Page 7 of 120"
N_OF_M_RE = re.compile(r"^\s*\d{1,4}\s*(?:of|/)\s*\d{1,4}\s*$", re.I)                       # "7 of 120", "7/120"
BARE_NUM_RE = re.compile(r"^\s*(\d{1,4})\s*$")


def extract_pages(pdf_path, layout=False, first=None, last=None):
    """One string per PDF page, using poppler's pdftotext (fast even on very large files)."""
    cmd = ["pdftotext", "-enc", "UTF-8"]
    if layout:
        cmd.append("-layout")
    if first:
        cmd += ["-f", str(first)]
    if last:
        cmd += ["-l", str(last)]
    cmd += [pdf_path, "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout.decode("utf-8", errors="replace")
    pages = out.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return pages


def _key(line, pdf_no=None, num_offset=None):
    """Comparable form of a line. ONLY digit runs equal to this page's own number are treated as variable, so body
    lines that differ in real data ('Rs 1037' vs 'Rs 1074', 'Table 3' vs 'Table 4') are never mistaken for a header."""
    own = set()
    if pdf_no is not None:
        own.add(pdf_no)
        if num_offset is not None:
            own.add(pdf_no + num_offset)
    t = re.sub(r"\d+", lambda m: "#" if int(m.group()) in own else m.group(), line.strip().lower())
    return re.sub(r"\s+", " ", t)


MAX_RUNNING_CHARS = 120      # real running headers/footers are short; anything longer is body text


def find_running_lines(pages, first=1, num_offset=None, edge=3, min_frac=0.4, min_pages=3):
    """Short lines that repeat, identically (apart from the page's own number), at the top/bottom of >=40% of pages."""
    if len(pages) < min_pages:
        return set()
    counts = Counter()
    for i, p in enumerate(pages):
        lines = [l for l in p.splitlines() if l.strip()]
        seen = set()
        for l in lines[:edge] + lines[-edge:]:
            if len(l.strip()) > MAX_RUNNING_CHARS:
                continue
            n = _key(l, first + i, num_offset)
            if n and n not in seen:
                seen.add(n)
                counts[n] += 1
    need = max(min_pages, int(min_frac * len(pages)))
    return {n for n, c in counts.items() if c >= need and len(n) > 3}


def find_page_number_offset(pages, first=1, edge=3, min_frac=0.4, min_pages=3):
    """Printed page numbers usually equal (PDF page + constant). Find that constant from lines that consist of a
    bare number at the top/bottom of pages. Returns None if no consistent pattern - then NO bare number is removed,
    because a bare number can also be real data (a year, a table value)."""
    if len(pages) < min_pages:
        return None
    votes = Counter()
    for i, p in enumerate(pages):
        lines = [l for l in p.splitlines() if l.strip()]
        offs = set()
        for l in lines[:edge] + lines[-edge:]:
            m = BARE_NUM_RE.match(l)
            if m:
                offs.add(int(m.group(1)) - (first + i))
        votes.update(offs)
    if not votes:
        return None
    off, cnt = votes.most_common(1)[0]
    return off if cnt >= max(min_pages, int(min_frac * len(pages))) else None


def strip_running(page, running, pdf_no=None, num_offset=None, edge=3):
    lines = page.splitlines()
    nonempty = [i for i, l in enumerate(lines) if l.strip()]
    edge_idx = set(nonempty[:edge] + nonempty[-edge:])
    kept, removed = [], []
    for i, l in enumerate(lines):
        if i in edge_idx and l.strip():
            m = BARE_NUM_RE.match(l)
            is_own_number = bool(m and num_offset is not None and pdf_no is not None and int(m.group(1)) == pdf_no + num_offset)
            if len(l.strip()) <= MAX_RUNNING_CHARS and (_key(l, pdf_no, num_offset) in running or PAGE_WORD_RE.match(l) or N_OF_M_RE.match(l) or is_own_number):
                removed.append(l.strip())
                continue
        kept.append(l)
    return "\n".join(kept), removed


def to_paragraphs(text):
    """Join wrapped lines into paragraphs; blank lines separate paragraphs."""
    paras, cur = [], []
    for raw in text.splitlines():
        line = re.sub(r"[ \t]+", " ", raw).strip()
        if not line:
            if cur:
                paras.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        paras.append(cur)
    out = []
    for lines in paras:
        s = lines[0]
        for nxt in lines[1:]:
            # keep a hyphen that ends a line (it may be a real compound such as human-elephant); no space if the next word is lowercase
            s += (nxt if (s.endswith("-") and nxt[:1].islower()) else " " + nxt)
        out.append(s)
    return "\n\n".join(out)


def to_lines(text):
    """--layout mode: keep one line per row and mark column gaps with ' | ', so tables stay readable."""
    out = []
    for raw in text.splitlines():
        line = re.sub(r"\s{2,}", " | ", raw.strip())
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out).strip()


def columnar_fraction(layout_text):
    """Share of non-empty lines containing a wide gap (3+ spaces) = text laid out in aligned columns (table, form, or a
    two-column page). Measured: ordinary prose pages ~0.00, a real form 0.62, a table 0.75."""
    lines = [l for l in layout_text.splitlines() if l.strip()]
    if len(lines) < 5:
        return 0.0
    return sum(len(re.findall(r"\S\s{3,}\S", l)) >= 1 for l in lines) / len(lines)


def digit_fraction(text):
    chars = [c for c in text if not c.isspace()]
    return (sum(c.isdigit() for c in chars) / len(chars)) if chars else 0.0


def ocr_page(pdf_path, page_no, dpi=300, lang="eng"):
    import pytesseract
    from PIL import Image
    with tempfile.TemporaryDirectory() as td:
        stem = os.path.join(td, "pg")
        subprocess.run(["pdftoppm", "-f", str(page_no), "-l", str(page_no), "-r", str(dpi), "-png", pdf_path, stem],
                       check=True, capture_output=True)
        png = [f for f in os.listdir(td) if f.endswith(".png")][0]
        return pytesseract.image_to_string(Image.open(os.path.join(td, png)), lang=lang)


def convert(pdf_path, doc_id, out_dir="data/clean", page_offset=0, ocr=False, layout=False, min_chars=40,
            table_digit_fraction=0.30, first=None, last=None):
    raw = extract_pages(pdf_path, layout=layout, first=first, last=last)
    lay = raw if layout else extract_pages(pdf_path, layout=True, first=first, last=last)   # only used to detect tables
    start = first or 1
    num_offset = find_page_number_offset(raw, first=start)
    running = find_running_lines(raw, first=start, num_offset=num_offset)
    report = {"doc_id": doc_id, "source_pdf": os.path.basename(pdf_path), "n_pages_read": len(raw),
              "page_numbering": "PDF viewer page number" if page_offset == 0 else "PDF page %+d" % page_offset,
              "running_lines_removed": sorted(running), "printed_number_offset": num_offset, "no_text_pages": [], "table_like_pages": [],
              "ocr_pages": [], "pages_written": 0, "chars_written": 0}
    parts = []
    for i, page in enumerate(raw):
        pdf_no = start + i
        label = pdf_no + page_offset
        cleaned, _ = strip_running(page, running, pdf_no, num_offset)
        text = to_lines(cleaned) if layout else to_paragraphs(cleaned)
        if len(text) < min_chars:
            report["no_text_pages"].append(label)
            if ocr:
                try:
                    o = (to_lines if layout else to_paragraphs)(strip_running(ocr_page(pdf_path, pdf_no), running, pdf_no, num_offset)[0])
                    if len(o) >= min_chars:
                        text = o
                        report["ocr_pages"].append(label)
                except Exception as e:  # tesseract missing, etc.: leave the page flagged
                    report.setdefault("ocr_errors", []).append("%s: %s" % (label, e))
            if len(text) < min_chars:
                continue
        if digit_fraction(text) > table_digit_fraction or (i < len(lay) and columnar_fraction(lay[i]) > 0.40):
            report["table_like_pages"].append(label)
        parts.append("<<<PAGE %s>>>\n%s\n" % (label, text))
        report["pages_written"] += 1
        report["chars_written"] += len(text)
    if len(running) > 4:
        report["warning"] = ("%d different running lines were removed - unusually many. Open the report and the .txt and make sure "
                             "nothing real was deleted." % len(running))
    os.makedirs(out_dir, exist_ok=True)
    txt_path = os.path.join(out_dir, doc_id + ".txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    with open(os.path.join(out_dir, doc_id + ".report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    report["txt_path"] = txt_path
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf")
    ap.add_argument("--doc-id", required=True, help="the Source_Inventory id, e.g. WII-001")
    ap.add_argument("--out", default="data/clean")
    ap.add_argument("--page-offset", type=int, default=0)
    ap.add_argument("--ocr", action="store_true", help="OCR pages that have no text layer (needs tesseract)")
    ap.add_argument("--layout", action="store_true", help="keep column layout (better for tables, worse for 2-column text)")
    ap.add_argument("--first", type=int)
    ap.add_argument("--last", type=int)
    a = ap.parse_args(argv)
    r = convert(a.pdf, a.doc_id, a.out, a.page_offset, a.ocr, a.layout, first=a.first, last=a.last)
    print("Wrote %s: %d pages, %d characters" % (r["txt_path"], r["pages_written"], r["chars_written"]))
    if r["running_lines_removed"]:
        print("Removed running header/footer lines (READ THIS LIST - make sure it is only headers/footers):", r["running_lines_removed"])
    if r.get("warning"):
        print("WARNING:", r["warning"])
    if r["no_text_pages"]:
        print("NO TEXT on pages %s%s" % (r["no_text_pages"], " (OCR recovered %s)" % r["ocr_pages"] if r["ocr_pages"] else
                                         " - scanned or figure pages; re-run with --ocr or transcribe by hand"))
    if r["table_like_pages"]:
        print("TABLE / COLUMN-LIKE pages (flattened text can scramble tables and two-column pages - re-run with --layout and check by hand):", r["table_like_pages"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
