"""Reads an uploaded CSV or XLSX into a header list and rows of strings.

Uses the csv module and openpyxl directly instead of pandas, which keeps the
serverless function smaller. Limits keep a single request fast and well under
Vercel's 4.5 MB request body limit.
"""
import csv
import io

MAX_BYTES = 4 * 1024 * 1024
MAX_ROWS = 5000
MAX_COLS = 60
MAX_CELL_CHARS = 2000


class FileError(ValueError):
    """A problem with the uploaded file, safe to show to the visitor."""


def _cell(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return s[:MAX_CELL_CHARS]


def _dedupe_headers(headers):
    seen, out = {}, []
    for i, h in enumerate(headers):
        h = _cell(h) or f"Column {i + 1}"
        if h in seen:
            seen[h] += 1
            h = f"{h} ({seen[h]})"
        else:
            seen[h] = 1
        out.append(h)
    return out


def read_table(filename: str, data: bytes):
    """Returns (headers, rows, truncated). rows are lists of strings aligned to
    headers. truncated is True when the file had more than MAX_ROWS rows."""
    if len(data) > MAX_BYTES:
        raise FileError("That file is larger than 4 MB. Split it into smaller files and try again.")
    name = (filename or "").lower()
    if name.endswith(".csv"):
        raw_rows = _read_csv(data)
    elif name.endswith(".xlsx"):
        raw_rows = _read_xlsx(data)
    else:
        raise FileError("Upload a .csv or .xlsx file.")

    if not raw_rows:
        raise FileError("The file is empty.")
    headers = _dedupe_headers(raw_rows[0][:MAX_COLS])
    body = [r for r in raw_rows[1:] if any(_cell(c) for c in r)]
    truncated = len(body) > MAX_ROWS
    rows = []
    for r in body[:MAX_ROWS]:
        cells = [_cell(c) for c in r[:MAX_COLS]]
        cells += [""] * (len(headers) - len(cells))
        rows.append(cells[: len(headers)])
    if not rows:
        raise FileError("The file has a header row but no data rows.")
    return headers, rows, truncated


def _read_csv(data: bytes):
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    return [row for row in csv.reader(io.StringIO(text), dialect)]


def _read_xlsx(data: bytes):
    from openpyxl import load_workbook

    try:
        wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:  # corrupt or not really an xlsx
        raise FileError("That .xlsx file couldn't be read.") from exc
    ws = wb.worksheets[0]
    rows = []
    for row in ws.iter_rows(values_only=True):
        rows.append(list(row))
        if len(rows) > MAX_ROWS + 1:
            break
    wb.close()
    return rows


def guess_text_column(headers, rows) -> str:
    """The column whose values look most like free text: longest average
    length among columns that are mostly non-numeric."""
    best, best_score = headers[0], -1.0
    sample = rows[:200]
    for i, h in enumerate(headers):
        values = [r[i] for r in sample if r[i]]
        if not values:
            continue
        numeric = sum(1 for v in values if v.replace(".", "", 1).replace("-", "", 1).isdigit())
        if numeric > len(values) / 2:
            continue
        score = sum(len(v) for v in values) / len(values)
        if any(k in h.lower() for k in ("review", "content", "text", "comment", "ulasan")):
            score *= 1.5
        if score > best_score:
            best, best_score = h, score
    return best
