import re


def normalize_text(text: str) -> str:
    """Replace common Unicode punctuation with ASCII equivalents.

    Matches the PyTorch TADA reference implementation.
    """
    substitutions = {
        # Quotes
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        # Dashes
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2010": "-",
        "\u2011": "-",
        # Ellipsis
        "\u2026": "...",
        # Misc punctuation
        "\u2039": "<",
        "\u203a": ">",
        "\u00ab": "<<",
        "\u00bb": ">>",
    }

    pattern = re.compile("|".join(re.escape(char) for char in substitutions))
    text = pattern.sub(lambda m: substitutions[m.group(0)], text)
    text = (
        text.replace("; ", ". ")
        .replace('"', "")
        .replace(":", ",")
        .replace("(", "")
        .replace(")", "")
        .replace("--", "-")
        .replace("-", ", ")
        .replace(",,", ",")
        .replace(" '", " ")
        .replace("' ", " ")
        .replace("  ", " ")
    )

    # Remove spaces before sentence-ending punctuation
    text = re.sub(r"\s+([.,?!])", r"\1", text)

    # Lowercase then capitalize after sentence-ending punctuation
    text = re.sub(
        r"([.!?]\s*)(\w)",
        lambda m: m.group(1) + m.group(2).upper(),
        text.lower(),
    )

    # Capitalize the very first character
    if text:
        text = text[0].upper() + text[1:]

    return text
