#!/usr/bin/env python3
"""Extract conservative code readability features for RQ0 classical baselines."""

from __future__ import annotations

import argparse
import json
import keyword
import logging
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


JAVA_KEYWORDS = {
    "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
    "class", "const", "continue", "default", "do", "double", "else", "enum",
    "extends", "final", "finally", "float", "for", "goto", "if", "implements",
    "import", "instanceof", "int", "interface", "long", "native", "new",
    "package", "private", "protected", "public", "return", "short", "static",
    "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
    "transient", "try", "void", "volatile", "while",
}

PYTHON_KEYWORDS = set(keyword.kwlist) | {"async", "await", "match", "case"}

C_FAMILY_KEYWORDS = {
    "alignas", "alignof", "asm", "auto", "bool", "break", "case", "catch",
    "char", "class", "const", "constexpr", "continue", "decltype", "default",
    "delete", "do", "double", "else", "enum", "explicit", "export", "extern",
    "false", "float", "for", "friend", "goto", "if", "inline", "int", "long",
    "namespace", "new", "noexcept", "nullptr", "operator", "private", "protected",
    "public", "register", "reinterpret_cast", "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "template", "this", "throw", "true", "try",
    "typedef", "typeid", "typename", "union", "unsigned", "using", "virtual", "void",
    "volatile", "while", "__device__", "__global__", "__host__", "__shared__",
}

TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[A-Za-z_$][A-Za-z0-9_$]*|\d+(?:\.\d+)?|==|!=|<=|>=|&&|\|\||[-+*/%=&|!<>^~?:;.,()[\]{}]'
)
IDENT_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*$")
LITERAL_RE = re.compile(r'^(?:"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|\d+(?:\.\d+)?)$')
OP_RE = re.compile(r"^(==|!=|<=|>=|&&|\|\||[-+*/%=&|!<>^~?:;.,()[\]{}])$")


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rq0_extract_features")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def entropy(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def language_keywords(language: str) -> set[str]:
    language = str(language).lower()
    if language in {"python", "py"}:
        return PYTHON_KEYWORDS
    if language in {"cuda", "c", "cpp", "c++"}:
        return C_FAMILY_KEYWORDS
    return JAVA_KEYWORDS


def strip_strings_and_comments(code: str, language: str = "java") -> str:
    code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
    code = re.sub(r"'(?:\\.|[^'\\])*'", "''", code)
    code = re.sub(r"/\*.*?\*/", " ", code, flags=re.DOTALL)
    code = re.sub(r"//.*", " ", code)
    if str(language).lower() in {"python", "py"}:
        code = re.sub(r"#.*", " ", code)
    return code


def extract_one(code: str, language: str = "java") -> dict[str, float]:
    code = "" if pd.isna(code) else str(code)
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = code.split("\n")
    nonempty = [line for line in lines if line.strip()]
    line_lengths = [len(line) for line in lines]
    nonempty_lengths = [len(line) for line in nonempty]
    blank_lines = len(lines) - len(nonempty)

    stripped_for_comments = lines
    comment_prefixes = ("#",) if str(language).lower() in {"python", "py"} else ("//",)
    line_comment_lines = sum(1 for line in stripped_for_comments if line.strip().startswith(comment_prefixes))
    block_comment_lines = sum(1 for line in stripped_for_comments if "/*" in line or "*" in line.strip()[:2] or "*/" in line)
    comment_lines = line_comment_lines + block_comment_lines

    indent_depths = []
    for line in nonempty:
        leading = len(line) - len(line.lstrip(" "))
        tabs = len(line) - len(line.lstrip("\t"))
        indent_depths.append(leading + tabs * 4)

    tokens = TOKEN_RE.findall(code)
    identifiers = [tok for tok in tokens if IDENT_RE.match(tok)]
    keyword_set = language_keywords(language)
    keywords = [tok for tok in identifiers if tok in keyword_set]
    non_keyword_identifiers = [tok for tok in identifiers if tok not in keyword_set]
    literals = [tok for tok in tokens if LITERAL_RE.match(tok)]
    operators = [tok for tok in tokens if OP_RE.match(tok)]

    code_no_comments = strip_strings_and_comments(code, language)
    control_keywords = re.findall(r"\b(if|for|while|case|catch|switch|&&|\|\|)\b", code_no_comments)
    cyclomatic_est = 1 + len(control_keywords)

    nesting_by_line = []
    depth = 0
    max_depth = 0
    for line in lines:
        depth = max(0, depth - line.count("}"))
        if line.strip():
            nesting_by_line.append(depth)
        depth += line.count("{")
        max_depth = max(max_depth, depth)

    distinct_ops = len(set(operators))
    distinct_operands = len(set(non_keyword_identifiers + literals))
    halstead_vocab = distinct_ops + distinct_operands
    halstead_len = len(operators) + len(non_keyword_identifiers) + len(literals)
    halstead_volume = 0.0 if halstead_vocab <= 1 else halstead_len * math.log2(halstead_vocab)

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else 0.0

    def std(values: list[float]) -> float:
        return float(np.std(values, ddof=0)) if values else 0.0

    token_lengths = [len(tok) for tok in tokens]
    identifier_lengths = [len(tok) for tok in non_keyword_identifiers]
    total_lines = max(len(lines), 1)
    return {
        "feature_loc_total": float(len(lines)),
        "feature_loc_nonempty": float(len(nonempty)),
        "feature_blank_line_ratio": float(blank_lines / total_lines),
        "feature_comment_line_ratio": float(comment_lines / total_lines),
        "feature_avg_line_length": mean(line_lengths),
        "feature_avg_nonempty_line_length": mean(nonempty_lengths),
        "feature_max_line_length": float(max(line_lengths) if line_lengths else 0),
        "feature_token_count": float(len(tokens)),
        "feature_avg_token_length": mean(token_lengths),
        "feature_token_entropy": entropy(tokens),
        "feature_identifier_count": float(len(non_keyword_identifiers)),
        "feature_identifier_unique_count": float(len(set(non_keyword_identifiers))),
        "feature_avg_identifier_length": mean(identifier_lengths),
        "feature_keyword_count": float(len(keywords)),
        "feature_operator_count": float(len(operators)),
        "feature_literal_count": float(len(literals)),
        "feature_indent_mean": mean(indent_depths),
        "feature_indent_max": float(max(indent_depths) if indent_depths else 0),
        "feature_indent_std": std(indent_depths),
        "feature_nesting_mean": mean(nesting_by_line),
        "feature_nesting_max": float(max_depth),
        "feature_cyclomatic_estimate": float(cyclomatic_est),
        "feature_halstead_distinct_operators": float(distinct_ops),
        "feature_halstead_distinct_operands": float(distinct_operands),
        "feature_halstead_vocabulary": float(halstead_vocab),
        "feature_halstead_length": float(halstead_len),
        "feature_halstead_volume": float(halstead_volume),
    }


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="experiments/rq0_viability/data/processed/pooled_313_processed.csv")
    parser.add_argument("--output", default="experiments/rq0_viability/data/processed/features_313.csv")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    root = repo_root / "experiments/rq0_viability"
    logger = setup_logger(root / "outputs/logs/extract_features.log")
    df = pd.read_csv(repo_root / args.dataset)
    logger.info("loaded dataset rows=%d", len(df))
    feature_rows = []
    for _, row in df.iterrows():
        features = extract_one(row["raw_code"], row.get("language", "java"))
        features["rq0_id"] = row["rq0_id"]
        features["dataset_name"] = row["dataset_name"]
        feature_rows.append(features)
    features_df = pd.DataFrame(feature_rows)
    output = repo_root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output, index=False)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str((repo_root / args.dataset).resolve()),
        "output": str(output.resolve()),
        "rows": int(len(features_df)),
        "feature_count": int(len([c for c in features_df.columns if c.startswith("feature_")])),
        "features": [c for c in features_df.columns if c.startswith("feature_")],
    }
    write_json(output.with_suffix(".manifest.json"), manifest)
    logger.info("wrote features=%s rows=%d feature_count=%d", output, len(features_df), manifest["feature_count"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
