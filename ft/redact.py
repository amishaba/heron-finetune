"""Redaction before anything leaves the machine (spec 14.5).

- Presidio (spaCy en_core_web_sm + pattern recognizers) replaces personal data with type placeholders
  such as <PERSON> or <PHONE_NUMBER>.
- Sender addresses become a stable salted hash token (`sender_7f3a9c`) plus the domain, so a model can
  learn "this sender matters" without seeing who it is. The salt lives in ~/.config/ft/salt.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from collections import Counter
from functools import cached_property
from pathlib import Path

ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "CREDIT_CARD",
    "IBAN_CODE",
    "US_SSN",
    "US_BANK_NUMBER",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "UK_NHS",
    "IP_ADDRESS",
    "CRYPTO",
    "URL",  # tracking links often carry personal identifiers
]
SPACY_MODEL = "en_core_web_sm"


def config_dir() -> Path:
    d = Path(os.environ.get("FT_CONFIG_DIR", Path.home() / ".config" / "ft"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _salt() -> bytes:
    p = config_dir() / "salt"
    if not p.exists():
        p.write_bytes(secrets.token_bytes(32))
        p.chmod(0o600)
    return p.read_bytes()


def sender_token(address: str) -> tuple[str, str]:
    """('sender_7f3a9c', 'example.com'): stable per address on this machine, not reversible elsewhere."""
    address = (address or "").strip().lower()
    domain = address.rsplit("@", 1)[-1] if "@" in address else ""
    digest = hashlib.sha256(_salt() + address.encode()).hexdigest()[:6]
    return f"sender_{digest}", domain


class Redactor:
    """Lazily builds Presidio (spaCy load takes ~1 s). Counts what it replaced, by entity type."""

    def __init__(self, score_threshold: float = 0.4):
        self.score_threshold = score_threshold
        self.counts: Counter[str] = Counter()

    @cached_property
    def _engines(self):
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        from presidio_anonymizer import AnonymizerEngine

        ignore = [
            "CARDINAL",
            "DATE",
            "EVENT",
            "FAC",
            "LANGUAGE",
            "LAW",
            "MONEY",
            "ORDINAL",
            "PERCENT",
            "PRODUCT",
            "QUANTITY",
            "TIME",
            "WORK_OF_ART",
        ]  # spaCy labels that aren't personal data
        nlp = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": SPACY_MODEL}],
                "ner_model_configuration": {"labels_to_ignore": ignore},
            }
        ).create_engine()
        return AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"]), AnonymizerEngine()

    def redact(self, text: str) -> str:
        if not text or not text.strip():
            return text
        from presidio_anonymizer.entities import OperatorConfig

        analyzer, anonymizer = self._engines
        results = analyzer.analyze(text=text, language="en", entities=ENTITIES, score_threshold=self.score_threshold)
        if not results:
            return text
        self.counts.update(r.entity_type for r in results)
        operators = {e: OperatorConfig("replace", {"new_value": f"<{e}>"}) for e in ENTITIES}
        return anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text
