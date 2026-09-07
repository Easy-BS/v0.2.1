# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 10:41:11 2025

@author: Xiguan Liang @SKKU
"""

# ./CALI_flow/run_graph.py

import os
import json
import argparse
import sys
from contextlib import redirect_stdout

from graph_config import define_graph

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--idf-in", dest="idf_in", required=False)
    p.add_argument("--idd", dest="idd_path", required=False)
    p.add_argument("--epw", dest="epw_path", required=False)
    p.add_argument("--model", required=False)
    args = p.parse_args()

    app = define_graph()

    init = {"user_input": args.prompt}
    if args.idf_in:
        init["idf_path"] = os.path.abspath(args.idf_in)
    if args.idd_path:
        init["idd_path"] = os.path.abspath(args.idd_path)
    if args.epw_path:
        init["epw_path"] = os.path.abspath(args.epw_path)
    if args.model:
        init["model"] = args.model

    with redirect_stdout(sys.stderr):
        final = app.invoke(init)

    print(json.dumps(final, ensure_ascii=False))

if __name__ == "__main__":
    main()
