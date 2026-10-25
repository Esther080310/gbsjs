#!/usr/bin/env python3
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from notice_pipeline.financial_features import prepare
if __name__=='__main__':print(json.dumps(prepare(Path(__file__).resolve().parents[1]),ensure_ascii=False,indent=2))
