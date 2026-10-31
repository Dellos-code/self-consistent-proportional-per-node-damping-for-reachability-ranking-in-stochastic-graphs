#!/bin/bash
# Post-processing once every confirmation panel is complete.
set -e
cd "$(dirname "$0")"
python3 mf_controls.py scale
python3 explore_rawP.py real,scale
python3 bench_runtime.py > logs/runtime.log 2>&1
python3 make_checkpoint_tables.py > logs/tables.log 2>&1
python3 write_checkpoint_report.py
echo FINISHED
