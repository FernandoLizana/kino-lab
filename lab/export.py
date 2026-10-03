from __future__ import annotations

import csv
import io
import json
from typing import Any


def table_csv(headers: list[str], rows: list[list[Any]]) -> bytes:
    buf = io.StringIO()
    buf.write("\ufeff")  # evita que Excel ejecute celdas
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(headers)
    for row in rows:
        writer.writerow(["'" + str(c) if str(c).startswith(("=", "+", "-", "@")) else c for c in row])
    return buf.getvalue().encode("utf-8")


def experiment_json(doc: dict) -> bytes:
    return json.dumps(doc, ensure_ascii=False, indent=2, default=str).encode("utf-8")


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def simple_pdf(title: str, paragraphs: list[str]) -> bytes:
    """PDF mínimo de texto. Sin ejecutar nada del contenido."""
    lines = [title, ""]
    for block in paragraphs:
        for raw in block.splitlines() or [""]:
            while raw:
                lines.append(raw[:90])
                raw = raw[90:]
            if not block:
                lines.append("")
    y = 800
    commands = ["BT", "/F1 11 Tf"]
    for line in lines[:60]:
        commands.append(f"1 0 0 1 40 {y} Tm ({_pdf_escape(line)}) Tj")
        y -= 14
    commands.append("ET")
    stream = "\n".join(commands).encode("latin-1", errors="replace")
    objs = []
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append(
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>"
    )
    objs.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer << /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    )
    return bytes(out)


def report_pdf(title: str, sections: dict[str, str]) -> bytes:
    paras = []
    for heading, body in sections.items():
        paras.append(heading.upper())
        paras.append(body)
        paras.append("")
    return simple_pdf(title, paras)
