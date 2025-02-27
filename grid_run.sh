#!/bin/bash

outfile=~/job_$OAR_JOB_ID.csv

echo "Outputting to ${outfile}"

uv pip install -r requirements.txt --system

parallel -j1 --joblog ./parallel.log --ssh oarsh --slf $OAR_NODEFILE "cd PGCNN && python3 data_efficiency.py" ::: {1..10} > ${outfile}
