import re
from typing import Any, Dict, Optional

LEGAL_SUFFIX_MAP = {
    "corp": "corporation",
    "corporation": "corporation",
    "inc": "incorporated",
    "incorporated": "incorporated",
    "ltd": "limited",
    "limited": "limited",
    "pvt": "private",
    "private": "private",
    "co": "company",
    "company": "company",
    "llc": "llc",
    "llp": "llp",
    "plc": "plc",
    "sarl": "sarl",
    "sas": "sas",
    "sasu": "sasu",
    "sa": "sa",
    "sci": "sci",
    "eurl": "eurl",
    "snc": "snc",
    "ste": "societe",
    "praivet limited": "private limited",
    "praivet": "private",
    "elelpi": "llp",
    "limitted": "limited",
    "kampani": "company",
}

ADDR_ABBR_MAP = {
    "rd": "road",
    "st": "street",
    "dr": "drive",
    "ave": "avenue",
    "ln": "lane",
    "ct": "court",
    "cir": "circle",
    "blvd": "boulevard",
    "pkwy": "parkway",
    "hwy": "highway",
    "sq": "square",
    "ste": "suite",
    "apt": "apartment",
    "fl": "floor",
    "bldg": "building",
    "pl": "place",
    "terr": "terrace",
    "bd": "boulevard",
    "rte": "route",
    "bat": "batiment",
}

LEGAL_WORDS = set(LEGAL_SUFFIX_MAP.keys()) | set(LEGAL_SUFFIX_MAP.values())
GENERIC_STOPWORDS = {
    "the", "and", "for", "of", "in", "at", "on", "to", "from", "with", "by",
    "des", "les", "du", "de", "la", "le", "et", "en", "dans", "pour", "par",
    "sur", "france", "paris", "group", "groupe", "international", "services", "service",
    "india", "usa", "united", "states",
}
BLOCK_STOPWORDS = LEGAL_WORDS | GENERIC_STOPWORDS


def normalize_name(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    tokens = t.split()
    norm_tokens = [LEGAL_SUFFIX_MAP.get(tok, tok) for tok in tokens]
    return " ".join(norm_tokens)


def normalize_address(text: str) -> str:
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"[^\w\s]", " ", t)
    tokens = t.split()
    norm_tokens = [ADDR_ABBR_MAP.get(tok, tok) for tok in tokens]
    return " ".join(norm_tokens)


def extract_postal_code(address: str) -> Optional[str]:
    if not address:
        return None
    match = re.search(r"\b(\d{5,6})\b", address)
    return match.group(1) if match else None


def normalize_record(row) -> Dict[str, Any]:
    eid, name, addr, country = str(row[0]), str(row[1] or ""), str(row[2] or ""), str(row[3] or "")
    norm_n = normalize_name(name)
    norm_a = normalize_address(addr)
    tokens = set(norm_n.split())
    search_tokens = {tok for tok in tokens if len(tok) >= 3 and tok not in BLOCK_STOPWORDS}
    if not search_tokens:
        search_tokens = {tok for tok in tokens if len(tok) >= 3}
    return {
        "entity_id": eid,
        "name": name,
        "address": addr,
        "country": country,
        "norm_name": norm_n,
        "norm_address": norm_a,
        "postal_code": extract_postal_code(addr),
        "name_tokens": tokens,
        "search_tokens": search_tokens,
        "addr_tokens": set(norm_a.split()),
    }
