# Data Collection
This was the data collection code used to collect profile, timeline, follower and following data off the X (formerly Twitter) API.

## To Run
To download the full dataset.

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 main.py config.json full $dataset $active_users_file
```