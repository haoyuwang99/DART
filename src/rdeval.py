#!/usr/bin/env python3
"""CLI entry for the DART evaluation grid --- thin wrapper over dart.eval. Run from the repo root:
  .venv311/bin/python src/rdeval.py <model_id> [datasets=agentdojo,mt] [n]   # append records
  .venv311/bin/python src/rdeval.py --report [records.jsonl]                 # render the tables
"""
from dart.eval import main

if __name__ == "__main__":
    main()
