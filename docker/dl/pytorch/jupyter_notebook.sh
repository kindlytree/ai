#   - jupyter notebook --generate-config
#     - Writing default config to: /root/.jupyter/jupyter_notebook_config.py
#     - c.NotebookApp.ip = '*'
#     - c.NotebookApp.port = 8888
#     - c.NotebookApp.notebook_dir = '/home/kindlytree'

#!/bin/bash
jupyter notebook --generate-config
echo "c.NotebookApp.ip = '*'" >> /root/.jupyter/jupyter_notebook_config.py
echo 'c.NotebookApp.port = 1234' >> /root/.jupyter/jupyter_notebook_config.py
echo "c.NotebookApp.notebook_dir = '/home/kindlytree'"  >> /root/.jupyter/jupyter_notebook_config.py
jupyter notebook password
# jupyter notebook --allow-root