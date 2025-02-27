#!/bin/bash

oarsub -l host=20 -p "cluster='gros'" ./grid_run.sh
