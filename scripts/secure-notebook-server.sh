#!/bin/bash
# Configure Jupyter notebook server to use password authentication
# Make sure Conda environment is active as will assume it is later
[ -z "$CONDA_PREFIX" ] && echo "Need to have Conda environment activated." && exit 1
if [ "$#" -gt 2 ]; then
    echo "Usage: bash secure-notebook-server.sh [jupyter-path] [open-ssl-config-path]"
    exit 1
fi
# If specified read Jupyter directory from passed argument
JUPYTER_DIR=${1:-"$HOME/.jupyter"}
# If specified read OpenSSL config file path from passed argument
# This is needed due to bug in how Conda handles config path
export OPENSSL_CONF=${2:-"$CONDA_PREFIX/ssl/openssl.cnf"}
SEPARATOR="=================================================================\n"
# Create default config file if one does not already exist
if [ ! -f "$JUPYTER_DIR/jupyter_notebook_config.py" ]; then
  echo "No existing notebook configuration file found, creating new one ..."
  printf $SEPARATOR
  jupyter notebook --generate-config
  printf $SEPARATOR
  echo "... notebook configuration file created."
fi
# Get user to enter notebook server password
echo "Getting notebook server password hash. Enter password when prompted ..."
printf $SEPARATOR
HASH=$(python -c "from jupyter_server.auth import passwd; print(passwd());")
printf $SEPARATOR
echo "... got password hash."
# Generate self-signed OpenSSL certificate and key file
echo "Creating certificate file ..."
printf $SEPARATOR
CERT_INFO="/C=UK/ST=Scotland/L=Edinburgh/O=University of Edinburgh/OU=School of Informatics/CN=$USER/emailAddress=$USER@sms.ed.ac.uk"
openssl req \
    -x509 -nodes -days 365 \
    -subj "/C=UK/ST=Scotland/L=Edinburgh/O=University of Edinburgh/OU=School of Informatics/CN=$USER/emailAddress=$USER@sms.ed.ac.uk" \
    -newkey rsa:1024 -keyout "$JUPYTER_DIR/key.key" \
    -out "$JUPYTER_DIR/cert.pem"
printf $SEPARATOR
echo "... certificate created."
# Setting permissions on key file
chmod 600 "$JUPYTER_DIR/key.key"
# Add password hash and certificate + key file paths to config file
echo "Setting up configuration file..."
printf $SEPARATOR
echo "   adding password hash"
SRC_PSW="^#\?c\.NotebookApp\.password[ ]*=[ ]*u['"'"'"]\(sha1:[a-fA-F0-9]\+\)\?['"'"'"]"
DST_PSW="c.NotebookApp.password = u'$HASH'"
grep -q "c.NotebookApp.password" $JUPYTER_DIR/jupyter_notebook_config.py
if [ ! $? -eq 0 ]; then
  echo DST_PSW >> $JUPYTER_DIR/jupyter_notebook_config.py
else
  sed -i "s/$SRC_PSW/$DST_PSW/" $JUPYTER_DIR/jupyter_notebook_config.py
fi
echo "   adding certificate file path"
SRC_CRT="^#\?c\.NotebookApp\.certfile[ ]*=[ ]*u['"'"'"]\([^'"'"'"]+\)\?['"'"'"]"
DST_CRT="c.NotebookApp.certfile = u'$JUPYTER_DIR/cert.pem'"
grep -q "c.NotebookApp.certfile" $JUPYTER_DIR/jupyter_notebook_config.py
if [ ! $? -eq 0 ]; then
  echo DST_CRT >> $JUPYTER_DIR/jupyter_notebook_config.py
else
  sed -i "s|$SRC_CRT|$DST_CRT|" $JUPYTER_DIR/jupyter_notebook_config.py
fi
echo "   adding key file path"
SRC_KEY="^#\?c\.NotebookApp\.keyfile[ ]*=[ ]*u['"'"'"]\([^'"'"'"]+\)\?['"'"'"]"
DST_KEY="c.NotebookApp.keyfile = u'$JUPYTER_DIR/key.key'"
grep -q "c.NotebookApp.keyfile" $JUPYTER_DIR/jupyter_notebook_config.py
if [ ! $? -eq 0 ]; then
  echo DST_KEY >> $JUPYTER_DIR/jupyter_notebook_config.py
else
  sed -i "s|$SRC_KEY|$DST_KEY|" $JUPYTER_DIR/jupyter_notebook_config.py
fi
printf $SEPARATOR
echo "... finished setting up configuration file."
