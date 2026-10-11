from io import BytesIO

from pypdf import PdfWriter


def minimal_pdf(texts: list[str]) -> bytes:
    """Build a minimal multi-page text PDF in memory (test fixture only)."""
    total = len(texts)
    objects: list[bytes] = [b"<< /Type /Catalog /Pages 2 0 R >>"]
    kids = " ".join(f"{3 + i} 0 R" for i in range(total))
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {total} >>".encode())
    for i in range(total):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {3 + total + i} 0 R "
            f"/Resources << /Font << /F1 {3 + 2 * total + i} 0 R >> >> >>".encode()
        )
    for text in texts:
        escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode()
        objects.append(
            b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream"
        )
    for _ in texts:
        objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF"
    out += trailer.encode()
    return bytes(out)


def encrypted_pdf() -> bytes:
    """Build a password-encrypted PDF (text extraction requires a password)."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.encrypt("secret")
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()
