#!/bin/sh
pip install jupyter
echo -e '#!/bin/bash\nunset XDG_RUNTIME_DIR\njupyter notebook --ip $(hostname -f) --no-browser' > $VIRTUAL_ENV/bin/notebook.sh
chmod u+x $VIRTUAL_ENV/bin/notebook.sh

pip install ipykernel
python -m ipykernel install --user --name msarthur-hface --display-name "Python 3.7 Arthur hugging"

