#!/usr/bin/env python3
"""Normalize local CSMAR ZIP exports without dropping or imputing records."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from notice_pipeline.csmar_ingest import normalize_all

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / 'artifacts/followup/csmar')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    normalize_all(args.root, args.output_dir)
