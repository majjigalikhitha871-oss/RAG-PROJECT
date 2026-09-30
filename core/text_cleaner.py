import re


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHENATED_LINE_BREAK = re.compile(r"(?<=\w)-\s*\n\s*(?=\w)")


def clean_text(text: str) -> str:
    """Normalize extraction artifacts while retaining paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00ad", "")
    text = _CONTROL_CHARS.sub("", text)
    text = _HYPHENATED_LINE_BREAK.sub("", text)
    paragraphs = re.split(r"\n\s*\n+", text)

    cleaned = []
    for paragraph in paragraphs:
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in paragraph.split("\n")]
        value = " ".join(line for line in lines if line)
        if value:
            cleaned.append(value)
    return "\n\n".join(cleaned)