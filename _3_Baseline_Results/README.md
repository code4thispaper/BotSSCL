# Baseline Results

These are a collection of our baseline results copied from seperate baseline source codes, put into one place to compile and graph.

## To Note

Botometer has three variants where the percentage threshold that an account is a bot or not. We consider the 0.5, 0.6 and 0.7 threshold as the baselines.

DigitalDNA has two variants B3 and B6 in their implementation.

## To Run
```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 tabulate.py
python3 graph.py
```