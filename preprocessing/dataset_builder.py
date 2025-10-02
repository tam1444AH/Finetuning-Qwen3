# dataset_builder.py
import json
import re
import re as _re2
from datetime import datetime
import math
import sys
import csv
import warnings
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Callable

try:
    import pandas as pd
except ImportError:
    pd = None
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutTimeout

# persistent decisions (optional)
_DECISION_CACHE: Dict[str, bool] = {}
_DECISION_CACHE_PATH: Optional[Path] = None
_COMMENTISH_STARTS = ("//", "#", "--", ";", "/*", "*", "*/")

def _line_type(line: str) -> str:
    s = line.lstrip()
    if not s:
        return "blank"
    return "comment" if s.startswith(_COMMENTISH_STARTS) else "code"

def normalize_separation_policy(text: str) -> str:
    """
    Enforce:
      1) No leading blank lines.
      2) Code↔Code: no empty blank line between (just newline).
      3) Any transition with comments: at most one empty line between.
      4) Exactly one trailing '\n' at EOF.
    """
    lines = text.split("\n")

    # 1) drop leading blanks
    i = 0
    n = len(lines)
    while i < n and lines[i].strip() == "":
        i += 1

    out: list[str] = []
    prev_type: str | None = None

    while i < n:
        # accumulate blanks
        if lines[i].strip() == "":
            # count run of blanks
            j = i
            while j < n and lines[j].strip() == "":
                j += 1
            # if blanks go to EOF, stop (no trailing blanks)
            if j >= n:
                break
            next_type = _line_type(lines[j])
            # decide how many blank lines to keep between prev non-blank and next non-blank
            if prev_type == "code" and next_type == "code":
                keep_blanks = 0  # Rule 2
            else:
                keep_blanks = 1  # Rule 3 (at most one)
            out.extend([""] * keep_blanks)
            i = j
            continue

        # non-blank line
        lt = _line_type(lines[i])
        out.append(lines[i])
        prev_type = lt
        i += 1

    # 4) exactly one trailing newline
    return ("\n".join(out)).rstrip("\n") + "\n"

def _load_decision_cache(path: Optional[Path]):
    global _DECISION_CACHE, _DECISION_CACHE_PATH
    _DECISION_CACHE_PATH = path
    if path and path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    h = obj.get("sha1"); keep = obj.get("keep")
                    if isinstance(h, str) and isinstance(keep, bool):
                        _DECISION_CACHE[h] = keep
            print(f"[hybrid-agent] loaded {len(_DECISION_CACHE)} cached decisions from {path}")
        except Exception as e:
            warnings.warn(f"could not load decision cache: {e}")

def _append_decision_cache(h: str, keep: bool):
    global _DECISION_CACHE_PATH
    if _DECISION_CACHE_PATH:
        try:
            with _DECISION_CACHE_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"sha1": h, "keep": keep}) + "\n")
        except Exception:
            pass  # cache write errors are non-fatal

def _hash_txt(txt: str) -> str:
    return hashlib.sha1(txt.encode("utf-8", errors="ignore")).hexdigest()

def estimate_tokens(text: str, *, chars_per_token: float = 4.0) -> int:
    """Very rough token estimate; errs high when chars_per_token < true ratio."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / max(1e-6, chars_per_token)))

def split_text_by_token_budget(
    text: str,
    *,
    max_tokens: int,
    overlap_tokens: int = 64,
    chars_per_token: float = 4.0,
    prefer_line_breaks: bool = True
) -> List[str]:
    """
    Split text into chunks that each fit within max_tokens (roughly).
    Uses a char-window consistent with chars_per_token, adds small overlap to reduce boundary artifacts.
    """
    if estimate_tokens(text, chars_per_token=chars_per_token) <= max_tokens:
        return [text]

    # Convert token budgets to char budgets
    max_chars = int(max_tokens * chars_per_token)
    overlap_chars = int(overlap_tokens * chars_per_token)

    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(n, i + max_chars)
        chunk = text[i:end]

        if prefer_line_breaks and end < n:
            # try to backtrack to last newline to avoid cutting mid-line
            back = chunk.rfind("\n")
            if back >= int(0.5 * max_chars):  # don’t backtrack too far (keeps chunks reasonable)
                end = i + back + 1
                chunk = text[i:end]

        chunks.append(chunk)
        if end >= n:
            break
        i = max(0, end - overlap_chars)
    return chunks

# ---------- Progress helpers ----------
def _print_every(n: int, total: Optional[int], label: str):
    now = datetime.now().strftime("%H:%M:%S")
    if total:
        pct = (n / total) * 100.0
        print(f"[{label}] {now} processed {n}/{total} ({pct:.1f}%)")
    else:
        print(f"[{label}] {now} processed {n}")


# ---------- Lazy import agent (now supports batching + progress) ----------
def _lazy_import_policy():
    try:
        from preprocessing.comment_policy_agent import decide_keep_drop_batch
        return decide_keep_drop_batch
    except Exception as e:
        warnings.warn(f"Hybrid comment agent unavailable ({e}). Falling back to heuristic batch.")
        def heuristic_batch(items: List[Dict]) -> List[bool]:
            outs = []
            for it in items:
                txt = it["comment_text"]
                lower = txt.lower()
                if any(k in lower for k in ["copyright", "license", "warranty", "apache", "mit", "bsd", "gnu"]):
                    outs.append(False); continue
                if len(txt) > 1000 and "http" in lower and "@" in txt:
                    outs.append(False); continue
                if any(k in lower for k in ["args", "parameters", "returns", "example", "usage", "invariant",
                                            "precondition", "postcondition", "proof", "spec"]):
                    outs.append(True); continue
                outs.append(len(txt) < 500)
            return outs
        return heuristic_batch

# ---------- Normalization & IO ----------
def read_text_normalized(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    try:
        data = path.read_bytes()
        text = data.decode('utf-8', errors='replace')
        # normalize newlines (real \r,\r\n -> \n)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        return text
    except Exception:
        return None

def prepend_root(p: Path, root_dir: Optional[Path]) -> Path:
    if not root_dir:
        return p
    try:
        # If absolute, leave as-is; else join root
        return p if p.is_absolute() else (root_dir / p)
    except Exception:
        return p

# ---------- Language guess ----------
EXT_LANG = {
    '.py': 'python', '.js': 'javascript', '.ts': 'typescript', '.java': 'java',
    '.c': 'c', '.h': 'c', '.hpp': 'cpp', '.cpp': 'cpp', '.cc': 'cpp',
    '.go': 'go', '.rs': 'rust', '.rb': 'ruby', '.php': 'php', '.sh': 'shell',
    '.bash': 'shell', '.zsh': 'shell', '.ps1': 'powershell', '.sql': 'sql',
    '.css': 'css', '.scss': 'css', '.md': 'markdown', '.txt': 'text',
    '.yaml': 'yaml', '.yml': 'yaml', '.cry': 'cryptol', '.saw': 'saw',
    '.lean': 'lean', '.hs': 'haskell',
}
def guess_lang_from_ext(path: Path) -> str:
    return EXT_LANG.get(path.suffix.lower(), 'text')

# ---------- Comment extraction/stripping ----------
BLOCK_C_STYLES = [
    (re.compile(r'/\*.*?\*/', re.S), '/*', '*/'),
]

LINE_COMMENT_PATTERNS = [
    (re.compile(r'(^|\s)//[^\n]*', re.M), '//'),
    (re.compile(r'(^|\s)#(?!\!).*', re.M), '#'),
    (re.compile(r'(^|\s)--[^\n]*', re.M), '--'),
    (re.compile(r'(^|\s);[^\n]*', re.M), ';'),
]

def extract_comments_raw(text: str) -> List[Tuple[str, Tuple[int,int], str]]:
    """Return list of (comment_text, (start,end), kind) spans (kind in {'block','//','#','--',';'})."""
    spans = []
    for rx, start_tok, end_tok in BLOCK_C_STYLES:
        for m in rx.finditer(text):
            spans.append((m.group(0), (m.start(), m.end()), 'block'))
    for rx, tok in LINE_COMMENT_PATTERNS:
        k = tok
        for m in rx.finditer(text):
            spans.append((m.group(0), (m.start(), m.end()), k))
    spans.sort(key=lambda x: x[1][0])
    return spans

def group_consecutive_slashslash(text: str, spans: List[Tuple[str, Tuple[int,int], str]]) -> List[Tuple[str, Tuple[int,int], str]]:
    """Group consecutive '//' line comments touching line-by-line into a single multi-line comment span."""
    grouped = []
    i = 0
    while i < len(spans):
        txt, (s, e), kind = spans[i]
        if kind != '//':
            grouped.append((txt, (s, e), kind))
            i += 1
            continue
        # Start a group of //
        start, end = s, e
        chunk = [txt]
        i += 1
        while i < len(spans) and spans[i][2] == '//':
            s2, e2 = spans[i][1]
            inter = text[end:s2]
            if set(inter) <= set([' ', '\t', '\n']):  # no code between
                end = e2
                chunk.append(spans[i][0])
                i += 1
            else:
                break
        grouped_txt = ''.join(chunk)
        grouped.append((grouped_txt, (start, end), '//'))
    grouped.sort(key=lambda x: x[1][0])
    return grouped

def extract_comments(text: str) -> List[Tuple[str, Tuple[int,int], str]]:
    raw = extract_comments_raw(text)
    return group_consecutive_slashslash(text, raw)

def strip_comments(text: str) -> Tuple[str, List[Tuple[str, Tuple[int,int], str]]]:
    """Remove comments and return (code_without_comments, list_of_removed (text, (s,e), kind))."""
    spans = extract_comments(text)
    if not spans:
        return text, []
    out = []
    removed = []
    idx = 0
    for ctext, (s, e), kind in spans:
        if s > idx:
            out.append(text[idx:s])
        removed.append((ctext, (s, e), kind))
        newlines = ctext.count('\n')
        out.append('\n' * newlines)
        idx = e
    out.append(text[idx:])
    return ''.join(out), removed

# ---------- Metrics ----------
import re as _re
HEXBYTE_RE = _re.compile(r'\b0x[0-9a-fA-F]+\b')
HEXNUM_RE = _re.compile(r'\b[0-9a-fA-F]{8,}\b')

def compute_basic_metrics(text: str) -> Dict[str, float]:
    lines = text.split('\n')
    n_lines = len(lines)
    n_bytes = len(text.encode('utf-8', errors='ignore'))
    avg_line_len = (sum(len(l) for l in lines) / max(1, n_lines))
    max_line_len = max((len(l) for l in lines), default=0)
    non_ascii_ratio = sum(1 for ch in text if ord(ch) > 127) / max(1, len(text))
    binary_like = int('\x00' in text)
    enc_base64_hits = len(_re.findall(r'\b[A-Za-z0-9+/=]{24,}\b', text))
    enc_hexbytes_hits = len(HEXBYTE_RE.findall(text))
    enc_unicode_hits = len(_re.findall(r'\\u[0-9a-fA-F]{4}', text))
    enc_total_matched = enc_base64_hits + enc_hexbytes_hits + enc_unicode_hits
    enc_max_run = max([len(m.group(0)) for m in _re.finditer(r'[A-Za-z0-9+/=]{1,}', text)] or [0])
    enc_fraction = enc_total_matched / max(1, len(text))
    lang_tokens = _re.findall(r'[A-Za-z_][A-Za-z0-9_]*', text)
    num_tokens_lang = len(lang_tokens)
    hexnum_ratio = len(HEXNUM_RE.findall(text)) / max(1, num_tokens_lang)
    num_tokens_model = len(_re.findall(r'\S+', text))
    # simple character-shingle count as proxy
    k = 5
    shingles = set()
    for line in lines:
        line = line.strip()
        for i in range(max(0, len(line) - k + 1)):
            shingles.add(line[i:i+k])
    num_shingles = len(shingles)
    return {
        'bytes': n_bytes, 'lines': n_lines, 'avg_line_len': avg_line_len, 'max_line_len': max_line_len,
        'non_ascii_ratio': non_ascii_ratio, 'binary_like': binary_like,
        'enc_total_matched': enc_total_matched, 'enc_max_run': enc_max_run, 'enc_fraction': enc_fraction,
        'enc_hits_base64': enc_base64_hits, 'enc_hits_hexbytes': enc_hexbytes_hits, 'enc_hits_unicode': enc_unicode_hits,
        'num_tokens_lang': num_tokens_lang, 'k_shingle': 5, 'num_shingles': num_shingles, 'hexnum_ratio': hexnum_ratio,
        'num_tokens_model': num_tokens_model,
    }

# ---------- Comment hashing & index ----------
def hash_comment(txt: str) -> str:
    return hashlib.sha1(txt.encode('utf-8', errors='ignore')).hexdigest()

_DECISION_CACHE: Dict[str, bool] = {}

def decide_batch(
    file_path: str,
    code_no_comments: str,
    spans: List[Tuple[str, Tuple[int,int], str]],
    *,
    batch_size: int = 10,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    agent_timeout_s: int = 60,
    max_comment_len: int = 4000,
) -> List[bool]:
    """
    Decide keep/drop using comment_policy_agent.decide_keep_drop_batch with:
      - per-batch timeouts
      - one retry with halved batch size
      - fallback heuristic if agent still fails
      - text truncation to avoid pathological inputs
      - persistent cache to skip repeat work
    """
    agent = _lazy_import_policy()

    # Prebuild undecided items, honoring cache and truncation
    undecided = []
    decisions: List[Optional[bool]] = [None]*len(spans)  # type: ignore
    for idx, (ctext, (s, e), kind) in enumerate(spans):
        h = _hash_txt(ctext)
        if h in _DECISION_CACHE:
            decisions[idx] = _DECISION_CACHE[h]
        else:
            # truncate long comments *only for agent input*; decision is still stored by hash of full text
            csend = ctext if len(ctext) <= max_comment_len else (ctext[:max_comment_len] + "\n/*...truncated...*/")
            undecided.append({"i": idx, "comment_text": csend, "file_path": file_path, "code_context": code_no_comments, "full_hash": h})

    processed = 0
    total = len(undecided)

    def _run_agent(items):
        # run agent in a thread so we can enforce a timeout
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(agent, items)
            return fut.result(timeout=agent_timeout_s)

    i = 0
    current_batch = batch_size if batch_size > 0 else 1
    while i < total:
        batch = undecided[i:i+current_batch]
        try:
            judged = _run_agent(batch)
            if len(judged) != len(batch):
                warnings.warn(f"Agent returned {len(judged)} decisions for batch of {len(batch)}. Aligning by min length.")
            upto = min(len(judged), len(batch))
            for j in range(upto):
                idx_global = batch[j]["i"]
                keep = bool(judged[j])
                decisions[idx_global] = keep
                _DECISION_CACHE[batch[j]["full_hash"]] = keep
                _append_decision_cache(batch[j]["full_hash"], keep)
            processed += upto
            if progress_cb:
                progress_cb(processed, total)
            i += current_batch
            # if successful with reduced size earlier, we can try to step back up a bit
            if current_batch < batch_size:
                current_batch = min(batch_size, current_batch * 2)
        except _FutTimeout:
            warnings.warn(f"[hybrid-agent] timeout after {agent_timeout_s}s on {file_path} batch starting at undecided[{i}] size={current_batch}. Retrying with half batch.")
            # one retry with half batch size
            if current_batch > 1:
                current_batch = max(1, current_batch // 2)
                continue
            # fallback heuristic for single item
            idx_global = batch[0]["i"]
            keep = len(spans[idx_global][0]) < 500
            decisions[idx_global] = keep
            _DECISION_CACHE[batch[0]["full_hash"]] = keep
            _append_decision_cache(batch[0]["full_hash"], keep)
            processed += 1
            if progress_cb:
                progress_cb(processed, total)
            i += 1
        except Exception as e:
            warnings.warn(f"[hybrid-agent] error on {file_path} batch starting at {i}: {e}. Falling back to heuristic for this batch.")
            # heuristic fallback for this batch
            upto = len(batch)
            for j in range(upto):
                idx_global = batch[j]["i"]
                keep = len(spans[idx_global][0]) < 500
                decisions[idx_global] = keep
                _DECISION_CACHE[batch[j]["full_hash"]] = keep
                _append_decision_cache(batch[j]["full_hash"], keep)
            processed += upto
            if progress_cb:
                progress_cb(processed, total)
            i += current_batch

    # fill any leftover None (belt-and-suspenders)
    for k in range(len(decisions)):
        if decisions[k] is None:
            decisions[k] = len(spans[k][0]) < 500

    return [bool(x) for x in decisions]  # type: ignore


def apply_hybrid_policy(
    original_text: str,
    file_path: str,
    code_no_comments: str,
    comments_index_fh,
    *,
    batch_size: int = 10,
    show_progress: bool = True,
    agent_timeout_s: int = 60,
    max_comment_len: int = 4000,
) -> str:
    spans = extract_comments(original_text)
    if not spans:
        return original_text

    def _cb(done: int, total: int):
        if show_progress:
            print(f"[hybrid-agent] {file_path}: decided {done}/{total} comment chunks")

    keeps = decide_batch(
        file_path,
        code_no_comments,
        spans,
        batch_size=batch_size,
        progress_cb=_cb if show_progress else None,
        agent_timeout_s=agent_timeout_s,
        max_comment_len=max_comment_len,
    )

    out = []
    idx = 0
    for (ctext, (s, e), kind), keep in zip(spans, keeps):
        if s > idx:
            out.append(original_text[idx:s])
        if keep:
            out.append(ctext)
        else:
            out.append('\n' * ctext.count('\n'))
        rec = {
            "filename": file_path, "span_start": s, "span_end": e, "kind": kind,
            "comment_sha1": hash_comment(ctext), "kept": bool(keep), "length": len(ctext)
        }
        comments_index_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        idx = e
    out.append(original_text[idx:])
    return ''.join(out)

def build_records_for_file(
    file_path: Path,
    variant: str,
    comments_index_fh=None,
    *,
    agent_batch_size: int = 10,
    show_agent_progress: bool = True,
    agent_timeout_s: int = 60,
    max_comment_len: int = 4000,
    # NEW: model context controls
    context_window_tokens: int = 4096,
    prompt_reserve_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
    chars_per_token: float = 4.0,
) -> Optional[List[Dict]]:
    """
    Returns a LIST of records (one per chunk) for this file+variant,
    or None if file unreadable.
    """
    text = read_text_normalized(file_path)
    if text is None:
        return None
    text = text.strip("\n") + "\n"
    lang = guess_lang_from_ext(file_path)
    code_wo, removed = strip_comments(text)

    if variant == 'with_comments':
        content = text
    elif variant == 'without_comments':
        content = code_wo
    elif variant == 'hybrid':
        if comments_index_fh is None:
            raise ValueError("comments_index_fh is required for hybrid variant")
        content = apply_hybrid_policy(
            text, str(file_path), code_wo, comments_index_fh,
            batch_size=agent_batch_size, show_progress=show_agent_progress,
            agent_timeout_s=agent_timeout_s, max_comment_len=max_comment_len
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")

    content = normalize_separation_policy(content)
    # --- chunk to fit model context (accounting for prompt tokens) ---
    max_usable_tokens = max(1, context_window_tokens - prompt_reserve_tokens)
    chunks = split_text_by_token_budget(
        content,
        max_tokens=max_usable_tokens,
        overlap_tokens=chunk_overlap_tokens,
        chars_per_token=chars_per_token,
        prefer_line_breaks=True
    )

    records: List[Dict] = []
    total_parts = len(chunks)
    for idx, chunk in enumerate(chunks, start=1):
        metrics = compute_basic_metrics(chunk)
        sha1 = hashlib.sha1(chunk.encode("utf-8", errors="ignore")).hexdigest()
        rec = {
            'filename': str(file_path),
            'lang': lang,
            'variant': variant,
            'content': chunk,
            'sha1': sha1,
            'nchars': len(chunk),
            # chunking metadata (handy for training/sampling)
            'chunk_idx': idx,
            'chunks_total': total_parts,
            'context_window_tokens': context_window_tokens,
            'prompt_reserve_tokens': prompt_reserve_tokens,
            **metrics,
        }
        records.append(rec)

    return records


def iter_source_files_from_metrics_csv(csv_path: Path, filename_col: str = 'filename',
                                       *, root_dir: Optional[Path] = None) -> Iterable[Path]:
    with csv_path.open('r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if filename_col in row and row[filename_col]:
                yield prepend_root(Path(row[filename_col]), root_dir)

def iter_source_files_from_jsonl(jsonl_path: Path, filename_field: str = 'filename',
                                 *, root_dir: Optional[Path] = None) -> Iterable[Path]:
    with jsonl_path.open('r') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if filename_field in obj and obj[filename_field]:
                    yield prepend_root(Path(obj[filename_field]), root_dir)
            except Exception:
                continue
def build_datasets(
    inputs: List[Path], out_dir: Path,
    variants: List[str] = ('with_comments','without_comments','hybrid'),
    save_jsonl: bool = True, save_parquet: bool = True,
    *, agent_batch_size: int = 10, show_agent_progress: bool = True,
    file_progress_every: int = 25,
    agent_timeout_s: int = 60, max_comment_len: int = 4000,
    decision_cache_path: Optional[Path] = None,
    # model context controls
    context_window_tokens: int = 4096,
    prompt_reserve_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
    chars_per_token: float = 4.0,
) -> Dict[str, List[Dict]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 1) load persistent agent decisions (safe no-op if path is None/missing)
    _load_decision_cache(decision_cache_path)

    results: Dict[str, List[Dict]] = {v: [] for v in variants}
    comments_index_fh = None
    if 'hybrid' in variants:
        comments_index_fh = (out_dir / 'comments_index.jsonl').open('w', encoding='utf-8')

    total_files = len(inputs)
    for idx_file, p in enumerate(inputs, start=1):
        # pretty, time-stamped progress line
        if file_progress_every and (idx_file % file_progress_every == 0 or idx_file == total_files):
            _print_every(idx_file, total_files, "dataset-builder")

        for v in variants:
            recs = build_records_for_file(
                p, v, comments_index_fh=comments_index_fh,
                agent_batch_size=agent_batch_size,
                show_agent_progress=show_agent_progress,
                agent_timeout_s=agent_timeout_s,
                max_comment_len=max_comment_len,
                context_window_tokens=context_window_tokens,
                prompt_reserve_tokens=prompt_reserve_tokens,
                chunk_overlap_tokens=chunk_overlap_tokens,
                chars_per_token=chars_per_token,
            )
            if recs:
                # 2) extend because we get a list of chunked records per file/variant
                results[v].extend(recs)

    if comments_index_fh is not None:
        comments_index_fh.close()

    # Always write the three JSONLs for fine-tuning (even if variants was narrowed)
    if save_jsonl:
        for v in ('with_comments','without_comments','hybrid'):
            if v not in results:
                results[v] = []
            jpath = out_dir / f'dataset_{v}.jsonl'
            with jpath.open('w', encoding='utf-8') as f:
                for r in results[v]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    if save_parquet and pd is not None:
        for v, recs in results.items():
            dpath = out_dir / f'dataset_{v}.parquet'
            df = pd.DataFrame(recs)
            df.to_parquet(dpath, index=False)
    elif save_parquet and pd is None:
        warnings.warn("pandas not installed — skipping parquet output.")

    return results

# ---------- Notebook-friendly wrapper ----------
def build_datasets_from_sources(
    *,
    metrics_csv: Optional[str] = None,
    jsonl: Optional[str] = None,
    filename_col: str = 'filename',
    filename_field: str = 'filename',
    out_dir: str = 'out_datasets',
    variants: str = 'with_comments,without_comments,hybrid',
    root_dir: Optional[str] = None,
    agent_batch_size: int = 10,
    show_agent_progress: bool = True,
    file_progress_every: int = 25,
    save_parquet: bool = True,
    agent_timeout_s: int = 60,
    max_comment_len: int = 4000,
    decision_cache_path: Optional[str] = None,
    # NEW: model context controls
    context_window_tokens: int = 4096,
    prompt_reserve_tokens: int = 512,
    chunk_overlap_tokens: int = 64,
    chars_per_token: float = 4.0,
) -> Dict[str, List[Dict]]:
    """
    High-level API for notebooks: pass metrics and/or jsonl, optional root_dir.
    """
    inputs: List[Path] = []
    root = Path(root_dir) if root_dir else None

    if metrics_csv:
        inputs.extend(iter_source_files_from_metrics_csv(Path(metrics_csv), filename_col, root_dir=root))
    if jsonl:
        inputs.extend(iter_source_files_from_jsonl(Path(jsonl), filename_field, root_dir=root))

    # unique inputs (preserve order)
    seen = set(); unique_inputs = []
    for p in inputs:
        if p not in seen:
            seen.add(p); unique_inputs.append(p)

    vlist = [v.strip() for v in variants.split(',') if v.strip()]
    return build_datasets(
        unique_inputs,
        Path(out_dir),
        variants=vlist,
        save_jsonl=True,
        save_parquet=save_parquet,
        agent_batch_size=agent_batch_size,
        show_agent_progress=show_agent_progress,
        file_progress_every=file_progress_every,
        agent_timeout_s=agent_timeout_s,
        max_comment_len=max_comment_len,
        decision_cache_path=Path(decision_cache_path) if decision_cache_path else None,
        context_window_tokens=context_window_tokens,
        prompt_reserve_tokens=prompt_reserve_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        chars_per_token=chars_per_token,
    )

# ---------- CLI ----------
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build 3 datasets: with comments, without comments, and hybrid (agent, batched, // grouped)")
    ap.add_argument('--metrics_csv', type=str, default='', help='CSV with a filename column (default: filename)')
    ap.add_argument('--jsonl', type=str, default='', help='JSONL with a filename field (default: filename)')
    ap.add_argument('--filename_col', type=str, default='filename')
    ap.add_argument('--filename_field', type=str, default='filename')
    ap.add_argument('--out_dir', type=str, default='out_datasets')
    ap.add_argument('--variants', type=str, default='with_comments,without_comments,hybrid')
    ap.add_argument('--root_dir', type=str, default='', help='Prepend this root to relative filenames when reading files')
    ap.add_argument('--agent_batch_size', type=int, default=10)
    ap.add_argument('--no_agent_progress', action='store_true')
    args = ap.parse_args()

    root = args.root_dir if args.root_dir else None
    results = build_datasets_from_sources(
        metrics_csv=args.metrics_csv or None,
        jsonl=args.jsonl or None,
        filename_col=args.filename_col,
        filename_field=args.filename_field,
        out_dir=args.out_dir,
        variants=args.variants,
        root_dir=root,
        agent_batch_size=args.agent_batch_size,
        show_agent_progress=not args.no_agent_progress,
    )
    print({k: len(v) for k, v in results.items()})

if __name__ == '__main__':
    main()
