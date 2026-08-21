#!/usr/bin/env python3
"""
Custom Clinical Text Preprocessing Pipeline for Medical Note Summarization.

This script implements domain-specific text normalization heuristics using pure Python
and regular expressions without relying on third-party NLP libraries (like spaCy/NLTK/Stanza).

Key Heuristics:
1. expand_clinical_abbreviations: Standardizes clinical acronyms and medical shorthand.
2. normalize_lab_values: Standardizes lab test names, numerical values, and units into structured tokens.
3. standardize_clinical_sections: Normalizes SOAP note headers and clinical section markers.
"""

import os
import re
import json
import argparse
from typing import List, Dict, Tuple


# Clinical Abbreviation Dictionary (Custom medical shorthand mapping)
CLINICAL_ABBREVIATIONS = {
    r"\bpt\b": "patient",
    r"\bpts\b": "patients",
    r"\bs/p\b": "status post",
    r"\bh/o\b": "history of",
    r"\bc/o\b": "complaining of",
    r"\bw/\b": "with",
    r"\bw/o\b": "without",
    r"\bdx\b": "diagnosis",
    r"\brx\b": "prescription",
    r"\bqd\b": "every day",
    r"\bbid\b": "twice a day",
    r"\btid\b": "three times a day",
    r"\bqid\b": "four times a day",
    r"\bprn\b": "as needed",
    r"\bnpo\b": "nothing by mouth",
    r"\bsob\b": "shortness of breath",
    r"\bcp\b": "chest pain",
    r"\bhtn\b": "hypertension",
    r"\bdm2\b": "type 2 diabetes mellitus",
    r"\bdm\b": "diabetes mellitus",
    r"\bcad\b": "coronary artery disease",
    r"\bchf\b": "congestive heart failure",
    r"\bckd\b": "chronic kidney disease",
    r"\baki\b": "acute kidney injury",
    r"\bafib\b": "atrial fibrillation",
    r"\bcabg\b": "coronary artery bypass graft",
    r"\bcopd\b": "chronic obstructive pulmonary disease",
    r"\bgerd\b": "gastroesophageal reflux disease",
    r"\bdvt\b": "deep vein thrombosis",
    r"\bpe\b": "pulmonary embolism",
    r"\buti\b": "urinary tract infection",
    r"\bra\b": "room air",
    r"\ble\b": "lower extremity",
    r"\bue\b": "upper extremity",
    r"\bpnd\b": "paroxysmal nocturnal dyspnea",
    r"\bjvd\b": "jugular venous distension",
    r"\beff?\b": "ejection fraction",
    r"\blkw\b": "last known well",
    r"\bcta\b": "computed tomography angiography",
    r"\bct\b": "computed tomography",
    r"\bmca\b": "middle cerebral artery",
    r"\bdes\b": "drug-eluting stent",
    r"\bpci\b": "percutaneous coronary intervention",
    r"\blad\b": "left anterior descending",
    r"\brll\b": "right lower lobe",
    r"\blll\b": "left lower lobe",
    r"\bns\b": "normal saline",
    r"\biv\b": "intravenous",
    r"\bpo\b": "by mouth",
    r"\byo\b": "year-old",
}

# Regex patterns for clinical lab tests and vital values
LAB_PATTERNS = [
    # Electrolytes and metabolic markers (e.g., K+ 5.2, Na 138, Cl 98, HCO3 22, BUN 42, Cr 2.1)
    (r"\b(K\+|Potassium)\s*([:=]?)\s*([0-9]+\.?[0-9]*)", r"LAB(name=K+, value=\3)"),
    (r"\b(Na|Sodium)\s*([:=]?)\s*([0-9]+\.?[0-9]*)", r"LAB(name=Na, value=\3)"),
    (r"\b(Cr|Creatinine)\s*([:=]?)\s*([0-9]+\.?[0-9]*)", r"LAB(name=Creatinine, value=\3)"),
    (r"\b(BUN)\s*([:=]?)\s*([0-9]+\.?[0-9]*)", r"LAB(name=BUN, value=\3)"),
    (r"\b(WBC)\s*([:=]?)\s*([0-9]+\.?[0-9]*)", r"LAB(name=WBC, value=\3)"),
    (r"\b(Hgb|Hemoglobin)\s*([:=]?)\s*([0-9]+\.?[0-9]*)", r"LAB(name=Hgb, value=\3)"),
    (r"\b(Plt|Platelets?)\s*([:=]?)\s*([0-9]+)", r"LAB(name=Platelets, value=\3)"),
    (r"\b(Glu|Glucose)\s*([:=]?)\s*([0-9]+)", r"LAB(name=Glucose, value=\3)"),
    (r"\b(Trop(?:\s*I)?|Troponin(?:\s*I)?)\s*([:=]?)\s*(<?[0-9]+\.?[0-9]*)", r"LAB(name=Troponin, value=\3)"),
    (r"\b(BNP)\s*([:=]?)\s*([0-9]+)", r"LAB(name=BNP, value=\3)"),
    (r"\b(Lipase)\s*([:=]?)\s*([0-9]+)", r"LAB(name=Lipase, value=\3)"),
    (r"\b(INR)\s*([:=]?)\s*([0-9]+\.?[0-9]*)", r"LAB(name=INR, value=\3)"),
    (r"\b(Lactate)\s*([:=]?)\s*([0-9]+\.?[0-9]*)", r"LAB(name=Lactate, value=\3)"),
    (r"\b(Procalcitonin)\s*([:=]?)\s*([0-9]+\.?[0-9]*)", r"LAB(name=Procalcitonin, value=\3)"),
    # Blood pressure patterns (e.g., BP 162/94 -> VITALS(BP=162/94 mmHg))
    (r"\bBP\s*([:=]?)\s*([0-9]{2,3}/[0-9]{2,3})", r"VITALS(BP=\2)"),
    # Oxygen saturation (e.g., SpO2 89% -> VITALS(SpO2=89%))
    (r"\bSpO2\s*([:=]?)\s*([0-9]{2,3}%?)", r"VITALS(SpO2=\2)"),
    # Heart rate (e.g., HR 98 -> VITALS(HR=98 bpm))
    (r"\bHR\s*([:=]?)\s*([0-9]{2,3})(?:\s*bpm)?", r"VITALS(HR=\2)"),
]

# Clinical Section Headers Normalization
SECTION_HEADERS = [
    (r"\bCHIEF COMPLAINT:\s*", "SECTION(CHIEF_COMPLAINT): "),
    (r"\bHPI:\s*", "SECTION(HISTORY_PRESENT_ILLNESS): "),
    (r"\bPMHX:\s*", "SECTION(PAST_MEDICAL_HISTORY): "),
    (r"\b(PHYSICAL EXAM(?: & VITALS)?|VITALS):\s*", "SECTION(PHYSICAL_EXAM): "),
    (r"\b(LABORATORY|LABS(?: & DIAGNOSTICS)?):\s*", "SECTION(LABS): "),
    (r"\b(IMAGING|CHEST X-RAY|EKG|CT ABDOMEN/PELVIS):\s*", r"SECTION(DIAGNOSTICS_\1): "),
    (r"\bASSESSMENT/PLAN:\s*", "SECTION(ASSESSMENT_PLAN): "),
]


def expand_clinical_abbreviations(text: str) -> str:
    """
    Custom Heuristic 1: Expands clinical abbreviations and medical shorthand using
    regex word-boundary pattern matching. Does not use external NLP packages.
    """
    cleaned = text
    for pattern, expansion in CLINICAL_ABBREVIATIONS.items():
        # Case-insensitive replacement preserving word boundary
        cleaned = re.sub(pattern, expansion, cleaned, flags=re.IGNORECASE)
    return cleaned


def normalize_lab_values(text: str) -> str:
    """
    Custom Heuristic 2: Finds unstructured laboratory values and vitals in clinical notes
    and normalizes them into structured tokens (e.g., LAB(name=K+, value=5.2)).
    """
    cleaned = text
    for pattern, replacement in LAB_PATTERNS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def standardize_clinical_sections(text: str) -> str:
    """
    Custom Heuristic 3: Normalizes SOAP note headers and unstructured clinical
    section dividers into consistent hierarchical tags.
    """
    cleaned = text
    for pattern, replacement in SECTION_HEADERS:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned


def clean_clinical_text(text: str) -> str:
    """
    Master pipeline function that sequentially applies all custom text normalization
    heuristics and formats whitespace and special characters.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 1. Expand medical abbreviations
    processed = expand_clinical_abbreviations(text)
    
    # 2. Normalize lab values and vitals
    processed = normalize_lab_values(processed)
    
    # 3. Standardize section headers
    processed = standardize_clinical_sections(processed)
    
    # 4. Normalize whitespace, line breaks, and punctuation artifacts
    processed = re.sub(r"\r\n|\r", "\n", processed)
    processed = re.sub(r"\n\s*\n+", "\n\n", processed)
    processed = re.sub(r"[ \t]+", " ", processed)
    processed = processed.strip()
    
    return processed


def preprocess_dataset(input_file: str, output_train: str, output_test: str, test_split_ratio: float = 0.25) -> Tuple[int, int]:
    """
    Reads raw clinical records, cleans input text and target summaries,
    and splits into train and test JSONL files.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at: {input_file}")
    
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    
    cleaned_records = []
    for item in records:
        raw_text = item.get("text", "")
        raw_summary = item.get("summary", "")
        
        cleaned_input = clean_clinical_text(raw_text)
        cleaned_target = clean_clinical_text(raw_summary)
        
        cleaned_records.append({
            "id": item.get("id", f"CLN-{len(cleaned_records):03d}"),
            "raw_text": raw_text,
            "raw_summary": raw_summary,
            "text": cleaned_input,
            "summary": cleaned_target,
            # Formatted model input with task instruction prefix
            "model_input": f"Summarize the following clinical note:\n{cleaned_input}"
        })
    
    # Determine split index
    total = len(cleaned_records)
    test_size = max(1, int(total * test_split_ratio))
    train_size = total - test_size
    
    train_data = cleaned_records[:train_size]
    test_data = cleaned_records[train_size:]
    
    os.makedirs(os.path.dirname(output_train), exist_ok=True)
    os.makedirs(os.path.dirname(output_test), exist_ok=True)
    
    with open(output_train, "w", encoding="utf-8") as f:
        for row in train_data:
            f.write(json.dumps(row) + "\n")
            
    with open(output_test, "w", encoding="utf-8") as f:
        for row in test_data:
            f.write(json.dumps(row) + "\n")
            
    # Also save combined cleaned data for reference
    combined_path = os.path.join(os.path.dirname(output_train), "cleaned_data.jsonl")
    with open(combined_path, "w", encoding="utf-8") as f:
        for row in cleaned_records:
            f.write(json.dumps(row) + "\n")
            
    print(f"[Preprocessing] Successfully processed {total} records:")
    print(f"  - Training samples saved to: {output_train} ({len(train_data)} records)")
    print(f"  - Testing samples saved to:  {output_test} ({len(test_data)} records)")
    
    return len(train_data), len(test_data)


def main():
    parser = argparse.ArgumentParser(description="Clinical Text Preprocessing Pipeline")
    parser.add_argument("--input", default="data/raw_data.jsonl", help="Path to raw input JSONL file")
    parser.add_argument("--train_out", default="data/cleaned_train.jsonl", help="Path for cleaned train output")
    parser.add_argument("--test_out", default="data/cleaned_test.jsonl", help="Path for cleaned test output")
    parser.add_argument("--test_split", type=float, default=0.25, help="Test split ratio (default: 0.25)")
    args = parser.parse_args()
    
    preprocess_dataset(args.input, args.train_out, args.test_out, args.test_split)


if __name__ == "__main__":
    main()
