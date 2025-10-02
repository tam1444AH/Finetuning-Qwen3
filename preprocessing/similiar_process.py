# similiar_process.py
# MinHash/LSH similarity over a candidate_df of absolute file paths.
# Saves CSV/Parquet/JSONL outputs and returns (df_files, df_pairs, similar_files).

import os
import sys
import re
import json
from typing import Dict, Tuple, Iterable, Optional, List

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

# ---------- defaults (configurable via run_from_dataframe args) ----------
DEFAULT_NUM_PERM      = 512
DEFAULT_K_SHINGLE     = 5
DEFAULT_LSH_THRESHOLD = 0.70
DEFAULT_TOP_N_PRINT   = 20
DEFAULT_OUT_DIR       = "minhash_outputs"

# ---------- helpers ----------
def tokenize(code: str) -> List[str]:
    # strip line & block comments
    code = re.sub(r'--.*?$|//.*?$|/\*.*?\*/', '', code, flags=re.S | re.M)

    token_pat = re.compile(r"""
        # ---------- literals ----------
        0x[0-9A-Fa-f]+            |   # hex
        0b[01]+                   |   # binary
        0o[0-7]+                  |   # octal
        \d+                       |   # decimal
        "(?:[^"\\]|\\.)*"        |   # strings (basic escapes)

        # ---------- identifiers ----------
        [A-Za-z_][A-Za-z_0-9']*   |   # allow prime in names (e.g., SubByte')

        # ---------- multi-char operators (order matters: longest first) ----------
        <<< | >>> | << | >>       |   # shifts/rotates
        \^\^                      |   # polynomial/exponent
        ::  | ->  | == | <= | >=  |   # comparisons/arrows
        !=  | <-  | \.\. | \.\.\. |   # not-equal, generator, ranges
        <\| | \|>                  |  # polynomial delimiters

        # ---------- single-char punctuation / operators ----------
        [{}()\[\];,:\.@!#\^+\-*/=<>\|`]
    """, re.X)

    toks = token_pat.findall(code)
    return toks

def shingles(tokens: List[str], k: int = 5) -> set:
    n = len(tokens)
    if n < k:
        return set()
    return {" ".join(tokens[i:i+k]) for i in range(n - k + 1)}

def jaccard(a: set, b: set) -> float:
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0

def to_minhash(sigset: Iterable[str], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for s in sigset:
        m.update(s.encode("utf-8"))
    return m

# ---------- helpers ----------
def _read_text_utf8_normalized(path: str) -> Optional[str]:
    """Open source file as UTF-8 (errors='replace') and normalize newlines to '\n'."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            t = f.read()
        # normalize newlines
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        return t
    except Exception as e:
        print(f"[warn] cannot read '{path}': {e}")
        return None

def _load_corpus_from_df(
    df: pd.DataFrame,
    filename_col: str = "filename",
    content_col: Optional[str] = None,
    root_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Build {filename -> content} from a candidate_df.
    If content_col provided, use it.
    Otherwise open files by joining root_dir (if given) + filename_col.
    """
    if filename_col not in df.columns:
        raise KeyError(f"'{filename_col}' column not found in DataFrame")

    corpus: Dict[str, str] = {}
    for i, row in df.iterrows():
        relname = row[filename_col]
        if not isinstance(relname, str) or not relname:
            print(f"[warn] row {i}: empty or non-string filename; skipping")
            continue

        text: Optional[str] = None
        if content_col and content_col in df.columns:
            raw = row[content_col]
            if isinstance(raw, str) and raw:
                text = raw.replace("\r\n", "\n").replace("\r", "\n")

        if text is None:
            path = os.path.join(root_dir, relname) if root_dir else relname
            text = _read_text_utf8_normalized(path)

        if text is None:
            print(f"[warn] row {i}: no content for '{relname}'; skipping")
            continue

        # KEY: keep relname as filename key (unchanged!)
        corpus[relname] = text

    return corpus


def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

# ---------- main entry ----------
def run_from_dataframe(
    candidate_df: pd.DataFrame,
    filename_col: str = "filename",
    content_col: Optional[str] = None,
    root_dir: Optional[str] = None,
    out_dir: str = DEFAULT_OUT_DIR,
    num_perm: int = DEFAULT_NUM_PERM,
    k_shingle: int = DEFAULT_K_SHINGLE,
    lsh_threshold: float = DEFAULT_LSH_THRESHOLD,
    top_n_print: int = DEFAULT_TOP_N_PRINT,
    save_parquet: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Execute MinHash/LSH + exact Jaccard over files referenced in candidate_df.

    Parameters
    ----------
    candidate_df : DataFrame with at least a 'filename' column of absolute paths.
    filename_col : Column name containing absolute file paths (default 'filename').
    content_col  : Optional column name containing already-loaded source text.
    out_dir      : Output directory for CSV/Parquet/JSONL.
    num_perm     : MinHash permutations.
    k_shingle    : k for k-shingles.
    lsh_threshold: LSH candidate threshold.
    top_n_print  : How many top pairs to preview in logs.
    save_parquet : Save parquet alongside CSV.

    Returns
    -------
    (df_files, df_pairs, similar_files)
    """
    _ensure_outdir(out_dir)
    print(f"[info] ==== Starting MinHash/LSH over DataFrame ====")
    print(f"[info] params: K_SHINGLE={k_shingle}, NUM_PERM={num_perm}, LSH_THRESHOLD={lsh_threshold}")

    # Build corpus from DataFrame (open UTF-8, normalize newlines)
    corpus = _load_corpus_from_df(
            candidate_df,
            filename_col=filename_col,
            content_col=content_col,
            root_dir=root_dir,
        )
    if not corpus:
        print("[fatal] No readable files from candidate_df; aborting.")
        return (pd.DataFrame(), pd.DataFrame(), [])

    print(f"[info] loaded {len(corpus)} files from candidate_df")

    # Build LSH and per-file signatures
    lsh = MinHashLSH(threshold=lsh_threshold, num_perm=num_perm)
    file_rows = []
    file_sigs: Dict[str, Tuple[set, MinHash]] = {}

    for path, text in corpus.items():
        toks = tokenize(text)
        S = shingles(toks, k=k_shingle)
        mh = to_minhash(S, num_perm)
        lsh.insert(path, mh)
        file_sigs[path] = (S, mh)

        hashvalues = mh.hashvalues.astype(np.uint64).tolist()
        file_rows.append({
            "filename": path,
            "num_tokens": len(toks),
            "num_shingles": len(S),
            "num_perm": num_perm,
            "k_shingle": k_shingle,
            "minhash_hashvalues": hashvalues,
        })

    df_files = pd.DataFrame(file_rows).sort_values("filename").reset_index(drop=True)
    print(f"[info] files indexed   : {len(df_files)}")

    # Candidate pairs: LSH + exact Jaccard
    pair_rows = []
    keys = list(corpus.keys())
    similar_files: List[str] = []

    for a in keys:
        S_a, mh_a = file_sigs[a]
        for b in lsh.query(mh_a):
            if b <= a:  # avoid (a,a) and dup pair directions by lexicographic order
                continue
            S_b, _ = file_sigs[b]
            s = jaccard(S_a, S_b)
            pair_rows.append({
                "a": a,
                "b": b,
                "jaccard": s,
                "a_shingles": len(S_a),
                "b_shingles": len(S_b),
                "union_shingles": len(S_a | S_b),
                "intersect_shingles": len(S_a & S_b),
            })
            if a not in similar_files:
                similar_files.append(a)
            if b not in similar_files:
                similar_files.append(b)

    df_pairs = pd.DataFrame(pair_rows).sort_values("jaccard", ascending=False).reset_index(drop=True)
    keep_t = lsh_threshold
    print(f"[diag] total candidate pairs: {len(df_pairs)}")
    print(f"[diag] pairs with jaccard >= {keep_t}: {(df_pairs['jaccard'] >= keep_t).sum()}")

    # ---------- save locally (CSV + optional Parquet) ----------
    files_csv      = os.path.join(out_dir, "minhash_files.csv")
    pairs_csv      = os.path.join(out_dir, "minhash_pairs.csv")
    files_parquet  = os.path.join(out_dir, "minhash_files.parquet")
    pairs_parquet  = os.path.join(out_dir, "minhash_pairs.parquet")
    sig_jsonl_path = os.path.join(out_dir, "minhash_signatures.jsonl")

    df_files.to_csv(files_csv, index=False)
    df_pairs.to_csv(pairs_csv, index=False)

    if save_parquet:
        try:
            df_files.to_parquet(files_parquet, index=False)
            df_pairs.to_parquet(pairs_parquet, index=False)
            print(f"[info] wrote CSV and Parquet to {out_dir}/")
        except Exception as e:
            print(f"[warn] parquet save failed ({e}); CSVs were still written to {out_dir}/")

    # JSONL of signatures for later reuse
    with open(sig_jsonl_path, "w", encoding="utf-8") as w:
        for _, row in df_files.iterrows():
            out = {
                "filename": row["filename"],
                "num_tokens": int(row["num_tokens"]),
                "num_shingles": int(row["num_shingles"]),
                "num_perm":     int(row["num_perm"]),
                "k_shingle":    int(row["k_shingle"]),
                "minhash_hashvalues": row["minhash_hashvalues"],
            }
            w.write(json.dumps(out) + "\n")

    # ---------- diagnostics ----------
    print()
    print("[info] ==== MinHash/LSH run summary ====")
    print(f"[info] files loaded  : {len(corpus)}")
    print(f"[info] files indexed : {len(df_files)}")

    zero_shingle = sum(1 for (_, (S, _)) in file_sigs.items() if not S)
    print(f"[info] files with 0 shingles (tokens < {k_shingle}): {zero_shingle}")

    print(f"[info] candidate pairs (from LSH) : {len(df_pairs)}")
    if not df_pairs.empty:
        for c in (0.6, 0.7, 0.8, 0.85, 0.9):
            n = int((df_pairs["jaccard"] >= c).sum())
            print(f"[info] pairs with Jaccard >= {c:.2f}: {n}")
        print(f"[info] avg Jaccard (candidates)  : {df_pairs['jaccard'].mean():.4f}")
        print(f"[info] max Jaccard               : {df_pairs['jaccard'].max():.4f}")
        print(f"[info] min Jaccard               : {df_pairs['jaccard'].min():.4f}")

        print("\n[info] top pairs:")
        cols = ["a", "b", "jaccard", "a_shingles", "b_shingles", "union_shingles", "intersect_shingles"]
        present = [c for c in cols if c in df_pairs.columns]
        print(df_pairs.sort_values("jaccard", ascending=False).head(top_n_print)[present].to_string(index=False))
    else:
        print("[warn] LSH returned zero candidates.")
        print("       Try K_SHINGLE=3–4, LSH_THRESHOLD=0.6–0.7, NUM_PERM ≥ 256.")

    print(f"[info] saved hash signatures JSONL: {sig_jsonl_path}")

    return df_files, df_pairs, similar_files

# ---------- example usage (not executed on import) ----------
if __name__ == "__main__":
    # Example: expect a parquet or csv with a 'filename' column of absolute paths.
    # python similiar_process.py /path/to/candidate_df.parquet
    import argparse
    p = argparse.ArgumentParser(description="Run MinHash/LSH over candidate_df of absolute file paths")
    p.add_argument("input", help="candidate_df .parquet or .csv with a 'filename' column")
    p.add_argument("--filename-col", default="filename")
    p.add_argument("--content-col", default=None)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    p.add_argument("--k-shingle", type=int, default=DEFAULT_K_SHINGLE)
    p.add_argument("--lsh-threshold", type=float, default=DEFAULT_LSH_THRESHOLD)
    p.add_argument("--no-parquet", action="store_true")
    args = p.parse_args()

    # Load df
    if args.input.lower().endswith(".parquet"):
        df = pd.read_parquet(args.input)
    elif args.input.lower().endswith(".csv"):
        df = pd.read_csv(args.input)
    else:
        print("[fatal] Unsupported input format. Use .parquet or .csv")
        sys.exit(1)

    run_from_dataframe(
        df,
        filename_col=args.filename_col,
        content_col=args.content_col if args.content_col else None,
        out_dir=args.out_dir,
        num_perm=args.num_perm,
        k_shingle=args.k_shingle,
        lsh_threshold=args.lsh_threshold,
        save_parquet=not args.no_parquet,
    )
