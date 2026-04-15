import re


class TextCleaner:
    _whitespace_pattern = re.compile(r"\s+")

    def clean(self, text: str) -> str:
        normalized = text.replace("\x00", " ").replace("\ufeff", " ")
        normalized = self._whitespace_pattern.sub(" ", normalized)
        return normalized.strip()
