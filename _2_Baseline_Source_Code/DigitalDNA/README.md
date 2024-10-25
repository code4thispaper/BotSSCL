# DigitalDNA

This baseline is based on the [Digital DNA Toolbox](https://github.com/WAFI-CNR/ddna-toolbox).

## To Run

To note, the Github installation does not work by default, and for the implementation to work, please comment out Line 114 in `venv/lib/python3.7/site-packages/digitaldna/lcs.py`.

```bash
python3 -m venv venv
source venv/bin/activate
git clone https://github.com/WAFI-CNR/ddna-toolbox
cd ddna-toolbox
git clone https://github.com/WAFI-CNR/glcr
pip3 install numpy 
pip3 install glcr/.
pip3 install .
pip3 install -r requirements
python3 ddna_preprocess.py
python3 model.py
```