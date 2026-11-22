"""Check explicit measurements and experiment identifiers against cited passages."""
import re
import unicodedata
from decimal import Decimal


_FACT = re.compile(
    r"(?<![a-z0-9.+-])(?:[+-]?\d+(?:\.\d+)?\s*(?:%|ms|seconds?|毫秒|秒|GB|MB)"
    r"|[+-]?\d+\.\d+|(?:bf|fp|int)(?:4|8|16|32|64)|[AHV]100"
    r"|(?:experiment|exp\.|实验)\s*#?\s*\d+[a-z]?|page\s+\d+|第\s*\d+\s*页)(?![a-z0-9])", re.I)
_PAGE = re.compile(r"(?:page\s+(\d+)|第\s*(\d+)\s*页)", re.I)
_EXPERIMENT = re.compile(r"(?:experiment|exp\.|实验)\s*#?\s*(\d+[a-z]?)", re.I)


def _facts(text):
    return {re.sub(r"\s+", "", m[0].casefold()) for m in _FACT.finditer(unicodedata.normalize("NFKC", text))}


def _clauses(text):
    return [clause.strip() for clause in re.split(r"[；;。!?！？]|(?<!\bExp)(?<!\bexp)\.(?=\s|$)", text)
            if clause.strip()]


def checkable_clauses(text, *, conditions="", limitation=False):
    clauses = [c for c in _clauses(text) if limitation or _facts(c) or re.search(
        r"\b(?:only|limited|unless|fails?|except)\b|仅限|局限|仅在|不适用", c, re.I)]
    if conditions.strip().casefold() not in {"", "unknown", "未知"}:
        clauses.extend(_clauses(conditions))
    return list(dict.fromkeys(clauses))


def _derived_ratio(proof, sentence):
    if proof.ratio is None:
        return set()
    operands = []
    units = []
    for literal in (proof.ratio.numerator, proof.ratio.denominator):
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(ms|s|seconds?|毫秒|秒)", literal, re.I)
        if not match or not re.search(r"(?<![\d.])" + re.escape(literal) + r"(?![a-z0-9])", sentence, re.I):
            return set()
        operands.append(Decimal(match[1]))
        units.append(match[2].casefold())
    if units[0] != units[1] or not operands[1]:
        return set()
    value = operands[0] / operands[1]
    return {m[1] for m in re.finditer(r"(?<![\d.])(\d+\.\d+)\s*(?:倍|×|x\b)", proof.clause)
            if round(value, len(m[1].split(".")[1])) == Decimal(m[1])}


def missing_fact_citations(clauses, proofs, spans, observations, experiment=""):
    """Only target-cited sources may ground a clause. Page numbers use PDF metadata."""
    supported = set()
    for proof in proofs:
        span = spans.get(proof.evidence_id)
        if span is None or proof.clause not in clauses or not proof.quote.strip() or proof.quote not in span.text:
            raise ValueError("事实核查引用了无效的目标分句、证据或逐字原文")
        # An exact substring can still clip the sign or leading digit of a measurement.
        start = r"(?<![a-zA-Z0-9.+-])" if proof.quote[0].isascii() and proof.quote[0].isalnum() else ""
        end = r"(?![a-zA-Z0-9])" if proof.quote[-1].isascii() and proof.quote[-1].isalnum() else ""
        if not re.search(start + re.escape(proof.quote) + end, span.text):
            continue
        expected = _facts(proof.clause)
        pages = list(_PAGE.finditer(proof.clause))
        if any(not span.page_start or not span.page_start <= int(m[1] or m[2]) <= (span.page_end or span.page_start)
               for m in pages):
            continue
        for page in pages:
            expected -= _facts(page[0])
        required_experiments = {m[1].casefold() for m in _EXPERIMENT.finditer(proof.clause + " " + experiment)}
        for sentence in _clauses(proof.quote):
            experiments = {m[1].casefold() for m in _EXPERIMENT.finditer(sentence + " " + span.section)}
            if expected <= (_facts(sentence) | _derived_ratio(proof, sentence)) and (not required_experiments or experiments == required_experiments):
                supported.add(proof.clause)
                break
    for obs in observations:
        if obs.status != "supports" or not any(s.paper_id == obs.paper_id and s.page_start
                and s.page_start <= obs.page <= (s.page_end or s.page_start) for s in spans.values()):
            continue
        facts = _facts(f"{obs.value} {obs.unit} {obs.conditions} {obs.observation}")
        for clause in clauses:
            if _facts(clause) and not _PAGE.search(clause) and (_facts(clause) | _facts(experiment)) <= facts:
                supported.add(clause)
    return [clause for clause in clauses if clause not in supported]
