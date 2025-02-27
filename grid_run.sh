#!/bin/bash

outfile=~/job_$OAR_JOB_ID.csv

echo "Outputting to ${outfile}"

uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

parallel -j1 --joblog ./parallel.log --ssh oarsh --slf $OAR_NODEFILE "cd PGCNN && source .venv/bin/activate && python3 data_efficiency.py" ::: {1..10} > ${outfile}
