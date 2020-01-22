# -*- coding: utf-8 -*-
# coding=utf-8
import streamlit as st
import os


ip_str = "52.15.166.93"

from PIL import Image
image = Image.open('haystac-logo-small.png')
st.sidebar.image(image, use_column_width=False)

from subprocess import check_output


stdoutdata = check_output(['curl', 'http://169.254.169.254/latest/meta-data/public-ipv4'])
ip_str = stdoutdata.decode("utf-8")


st.header("NLP Demos Control Panel")
st.write(f"External ip: {ip_str}")

if st.sidebar.button("Start Question Answering"):
  # os.system("pkill -f nlpdemop")
  # os.system("pkill -f streamlit")
  os.system("sudo kill -9 $(pgrep -f runner.py)")
  os.system("sudo systemctl stop nlpdemos.service")
  os.system('/home/ubuntu/swi/runlitwe.sh /home/ubuntu/qa runner.py')
  # st.write(f'<a href="http://{ip_str}:8501" target="_blank">Open Q&A</a>', unsafe_allow_html=True)

if st.sidebar.button("Start Extractive Summary"):
  # os.system("pkill -f nlpdemop")
  # os.system("pkill -f streamlit")
  os.system("sudo kill -9 $(pgrep -f runner.py)")
  os.system("sudo systemctl stop nlpdemos.service")
  os.system('/home/ubuntu/swi/runlitwe.sh /home/ubuntu/es es_runner.py')
  # st.write(f'<a href="http://{ip_str}:8501" target="_blank">Open Extractive summarization</a>', unsafe_allow_html=True)


if st.sidebar.button("Start NER"):
  # os.system("pkill -f nlpdemop")
  # os.system("pkill -f streamlit")
  os.system("sudo kill -9 $(pgrep -f runner.py)")
  os.system("sudo systemctl start nlpdemos.service")
  # os.system('/home/ubuntu/swi/runlitwe.sh /home/ubuntu/qa runner.py')
  # st.write(f'<a href="http://{ip_str}:8080" target="_blank">Open NER</a>', unsafe_allow_html=True)


try:
  stdoutdata = check_output('ps aux | grep -E "nlpdemo|runner.py" | grep -v grep', shell=True)
  st.subheader("Running processes:")
  processes_str = stdoutdata.decode("utf-8")
  st.write(f"<pre>{processes_str}</pre>", unsafe_allow_html=True)

  if  "nlpdemo-app.py" in processes_str:
    st.write(f'<a href="http://{ip_str}:8080" target="_blank">Open NER</a>', unsafe_allow_html=True)

  if  "run runner.py" in processes_str:
    st.write(f'<a href="http://{ip_str}:8501" target="_blank">Open Q&A</a>', unsafe_allow_html=True)

  if  "run es_runner.py" in processes_str:
    st.write(f'<a href="http://{ip_str}:8501" target="_blank">Open Extractive summarization</a>',
             unsafe_allow_html=True)
except:
  st.write("no processes.")

st.write(f'<hr>', unsafe_allow_html=True)

try:
  stdoutdata = check_output('nvidia-smi', shell=True)
  st.subheader("Nvidia status:")
  processes_str = stdoutdata.decode("utf-8")
  st.write(f"<pre>{processes_str}</pre>", unsafe_allow_html=True)

except:
  st.write("")
