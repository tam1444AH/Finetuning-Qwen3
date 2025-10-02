# ---------- metrics helpers ----------
import re, hashlib
from pathlib import Path

# Encoded-data (StarCoder-style)
_ENC_BASE64_RE   = re.compile(r"[A-Za-z0-9+/=\n]{64,}")
_ENC_HEXBYTES_RE = re.compile(r"(?:\b(?:0x|\\x)?[0-9A-Fa-f]{2}(?:,|\b\s*)){8,}")
_ENC_UNICODE_RE  = re.compile(r"(?:\\u[0-9A-Fa-f]{4}){8,}")

# Binary-like detector (control chars except \t \n \r)
_BINARY_CHUNK = re.compile(rb"[\x00-\x08\x0B\x0C\x0E-\x1F]")

# Long hex / long number
_HEX_LONG_RE = re.compile(r"0x[0-9A-Fa-f]{4,}")
_NUM_LONG_RE = re.compile(r"\b\d{6,}\b")

# Junk path
_JUNK_PATH_RE = re.compile(
    r"(?:^|/)(?:test|tests|example|examples|tmp|backup|backups|copy|copies|vendor|third_party|old)(?:/|$)",
    re.I
)

def normalize_newlines(s: str) -> str:
    return s.replace("\r\n", "\n").replace("\r", "\n")

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def looks_binary(text: str) -> bool:
    raw = text.encode("utf-8", errors="ignore")
    return bool(_BINARY_CHUNK.search(raw))

def line_stats(text: str) -> tuple[int, float, int]:
    t = normalize_newlines(text)
    lines = t.split("\n")
    if len(lines) == 1 and lines[0] == "":
        return 0, 0.0, 0
    total = 0
    maxlen = 0
    for ln in lines:
        L = len(ln)
        total += L
        if L > maxlen: maxlen = L
    return len(lines), (total / len(lines)), maxlen

def non_ascii_ratio(text: str) -> float:
    if not text: return 0.0
    non = sum(1 for ch in text if ord(ch) > 127)
    return non / len(text)

def encoded_data_metrics(text: str) -> dict:
    length = max(1, len(text))
    total_matched = 0
    max_run = 0
    hits = {"base64": 0, "hexbytes": 0, "unicode": 0}
    for name, pat in (("base64", _ENC_BASE64_RE),
                      ("hexbytes", _ENC_HEXBYTES_RE),
                      ("unicode", _ENC_UNICODE_RE)):
        for m in pat.finditer(text):
            span = m.end() - m.start()
            total_matched += span
            if span > max_run: max_run = span
            hits[name] += 1
    return {
        "enc_total_matched": total_matched,
        "enc_max_run": max_run,
        "enc_fraction": total_matched / length,
        "enc_hits_base64": hits["base64"],
        "enc_hits_hexbytes": hits["hexbytes"],
        "enc_hits_unicode": hits["unicode"],
    }

def hex_num_ratio(text: str, token_count_hint: int | None = None) -> float:
    long_hex = len(_HEX_LONG_RE.findall(text))
    long_num = len(_NUM_LONG_RE.findall(text))
    denom = max(1, token_count_hint or 0)
    return min(1.0, (long_hex + long_num) / denom)

# Default very-light Cryptol-ish tokenizer (you can pass your own)
_DEFAULT_TOKENIZER = re.compile(r"""
    0x[0-9A-Fa-f]+ | 0b[01]+ | 0o[0-7]+ | \d+ |
    "(?:[^"\\]|\\.)*" |
    [A-Za-z_][A-Za-z_0-9']* |
    <<<|>>>|<<|>>|\^\^|::|->|==|<=|>=|!=|<-|\.\.|\.\.\.|<\||\|> |
    [{}()\[\];,:\.@!#\^+\-*/=<>\|`]
""", re.X)

def default_tokenize(code: str) -> list[str]:
    # strip line/block comments first
    code = re.sub(r'--.*?$|//.*?$|/\*.*?\*/', '', code, flags=re.S|re.M)
    return _DEFAULT_TOKENIZER.findall(code)

# ---------- main metrics function ----------
def compute_file_metrics(
    filename: str,
    text: str,
    *,
    k_shingle: int = 5,
    lang_tokenize = None,              # callable: (str)->list[str]
    model_tokenizer = None             # HuggingFace tokenizer or any object with encode(add_special_tokens=False)
) -> dict:
    """
    Return a flat dict of per-file metrics suitable for a pandas DataFrame row.
    Does not make keep/drop decisions—just measures.
    """
    lang_tokenize = lang_tokenize or default_tokenize

    # Normalize once for stable hashing & line stats
    norm = normalize_newlines(text)
    sha1 = sha1_text(norm)

    # Bytes (encoded) and line stats
    n_bytes = len(norm.encode("utf-8", errors="ignore"))
    n_lines, avg_line_len, max_line_len = line_stats(norm)

    # Quick checks
    is_binary = looks_binary(norm)
    na_ratio  = non_ascii_ratio(norm)

    # Encoded data coverage
    enc = encoded_data_metrics(norm)

    # Language tokens & shingles
    toks = lang_tokenize(norm)
    num_tokens_lang = len(toks)
    num_shingles    = max(0, num_tokens_lang - k_shingle + 1)

    # Hex/long-number concentration (per token)
    hexnum_ratio = hex_num_ratio(norm, token_count_hint=num_tokens_lang)

    # Model tokens (optional)
    num_tokens_model = None
    if model_tokenizer is not None:
        try:
            num_tokens_model = len(model_tokenizer.encode(norm, add_special_tokens=False))
        except Exception:
            num_tokens_model = None

    # Junk path heuristic
    path_norm = filename.replace("\\", "/")
    junk_path = bool(_JUNK_PATH_RE.search(path_norm))

    return {
        # identity
        "filename": filename,
        "sha1": sha1,

        # size / lines
        "bytes": n_bytes,
        "lines": n_lines,
        "avg_line_len": round(avg_line_len, 2),
        "max_line_len": max_line_len,

        # content character stats
        "non_ascii_ratio": round(na_ratio, 6),
        "binary_like": is_binary,

        # encoded-data coverage
        **enc,  # enc_total_matched, enc_max_run, enc_fraction, enc_hits_*
        
        # token/shingle stats
        "num_tokens_lang": num_tokens_lang,
        "k_shingle": k_shingle,
        "num_shingles": num_shingles,
        "hexnum_ratio": round(hexnum_ratio, 6),

        # model tokens (optional)
        "num_tokens_model": num_tokens_model,

        # path heuristics
        "junk_path": junk_path,
    }

