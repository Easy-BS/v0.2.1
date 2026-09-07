# -*- coding: utf-8 -*-
"""
Created on Mon Sep 22 16:32:56 2025

@author: Xiguan Liang @SKKU
"""

# CALI_flow/nodes/cali_runtime_config.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

def load_runtime_config() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", required=False)
    args, _unknown = parser.parse_known_args()

    if not args.config:
        return {}

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config JSON not found: {cfg_path}")

    return json.loads(cfg_path.read_text(encoding="utf-8"))