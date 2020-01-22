#!/bin/bash

cd $1
source ./venv/bin/activate
exec -a NlpDemop streamlit run $2 &


