"""
Text cleaning. The trickiest part is that the same cleaning has to be
applied at training time AND at inference time, otherwise the feature
space drifts. We expose one function (preprocess_message) that both paths
call, so they can never disagree.
"""
import re
import pandas as pd

from src.config import COLUMNS
from src.logging_utils import get_logger

log = get_logger(__name__)


# --- Boilerplate patterns ---------------------------------------------------
# Customer-support template phrases in various languages. The starter code
# had these as a dict per language - we just flatten into one big OR pattern
# since we don't actually care which language matched.
_BOILERPLATE = re.compile("|".join([
    r"(?:Aspiegel|\*{5}\(PERSON\)) Customer Support team\,?",
    r"(?:Aspiegel|\*{5}\(PERSON\)) SE is a company incorporated under the laws of Ireland with its headquarters in Dublin, Ireland\.?",
    r"(?:Aspiegel|\*{5}\(PERSON\)) Kundenservice\,?",
    r"L'équipe d'assistance à la clientèle d'Aspiegel\,?",
    r"(?:Aspiegel|\*{5}\(PERSON\)) Soporte Servicio al Cliente\,?",
    r"Il tuo team ad (?:Aspiegel|\*{5}\(PERSON\))",
]), re.IGNORECASE)

# Patterns that mark the start of a quoted reply (so we can split chains)
_SPLIT_PATTERN = re.compile("|".join([
    r"From\s?:\s?xxxxx@xxxx\.com Sent\s?:.{30,70}Subject\s?:",
    r"On.{30,60}wrote:",
    r"\bRe\s?:|\bRE\s?:",
    r"\*{5}\(PERSON\) Support issue submit",
    r"\s?\*{5}\(PHONE\)\s*$",
]))

# All the junk patterns the starter pipeline used. Compiled once so we don't
# pay the regex-compile cost on every row.
_NOISE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(\br\s*:)|(\bre\s*:)",
    r"\[|\]",
    r"aspiegel support issue submit",
    r"\bnull\b|\bnan\b",
    r"(from :)|(subject :)|(sent :)",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    r"\b(jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b",
    r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\d{2}[:.]\d{2}",
    r"xxxxx@xxxx\.com|\*{5}\([a-z]+\)",
    r"dear ((customer)|(user))|^dear\b",
    r"\b(hello|hallo|hi|hi there|good morning)\b",
    # collapse all the various "thank you for..." phrases
    r"thank you( very much| very kindly| for (your )?(patience|cooperation|reply|response|contacting|availability|providing))?",
    r"sent from my huawei( cell)? phone",
    r"original message",
    r"customer support team",
    r"we apologize for the inconvenience",
    r"\d+",                  # drop bare numbers
    r"[^0-9a-zA-Z\s]+",      # punctuation
]]

_WS = re.compile(r"\s+")


def clean_text(text):
    """Apply the full noise-removal pipeline to one string."""
    if not isinstance(text, str) or not text:
        return ""
    s = text.lower()
    for p in _NOISE_PATTERNS:
        s = p.sub(" ", s)
    return _WS.sub(" ", s).strip()


def clean_summary(text):
    """Lighter cleaning for the ticket summary - we keep more signal here."""
    if not isinstance(text, str) or not text:
        return ""
    s = text.lower()
    s = re.sub(r"(sv\s*:)|(wg\s*:)|(ynt\s*:)|(fw(d)?\s*:)|(\br\s*:)|(\bre\s*:)", " ", s)
    s = re.sub(r"\[|\]", " ", s)
    return _WS.sub(" ", s).strip()


def deduplicate_interactions(df):
    """
    Tickets often contain quoted email chains where the same paragraph
    appears multiple times. We split on the quote-start markers and drop
    duplicate fragments within each ticket.
    """
    df = df.copy()
    deduped = [""] * len(df)

    for ticket_id, group in df.groupby(COLUMNS.ticket_id, sort=False):
        seen = set()
        for idx, ic in zip(group.index, group[COLUMNS.interaction_content]):
            if not isinstance(ic, str):
                continue
            parts = [p for p in _SPLIT_PATTERN.split(ic) if p is not None]
            kept = []
            for part in parts:
                part = _BOILERPLATE.sub("", part).strip()
                if part and part not in seen:
                    seen.add(part)
                    kept.append(part)
            deduped[idx] = " ".join(kept) if kept else ic

    df[COLUMNS.interaction_content] = deduped
    return df


def preprocess_dataframe(df):
    """Full training-time preprocessing - dedup, clean, build combined column."""
    df = deduplicate_interactions(df)
    df[COLUMNS.ticket_summary] = df[COLUMNS.ticket_summary].apply(clean_summary)
    df[COLUMNS.interaction_content] = df[COLUMNS.interaction_content].apply(clean_text)
    df["text_combined"] = (
        df[COLUMNS.ticket_summary].fillna("") + " " +
        df[COLUMNS.interaction_content].fillna("")
    ).str.strip()
    log.info("Preprocessed %d rows", len(df))
    return df


def preprocess_message(summary, content):
    """
    Single-message preprocessing for the inference path.
    IMPORTANT: this calls the same clean_* functions that the trainer uses
    row-by-row. If you change anything here you must change it for training.
    """
    return (clean_summary(summary) + " " + clean_text(content)).strip()
