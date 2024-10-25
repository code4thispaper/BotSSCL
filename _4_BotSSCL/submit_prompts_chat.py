from openai import OpenAI
import pandas as pd
import ujson
import os
import sys

def sample_bots(datasets):
    for d in datasets:
        df = pd.read_csv(f"Data/{d}-clean.csv", index_col=0)
        bots = df[df['label'] == 1]

        # Extract bots with tweets
        ids = []
        for id in bots['id']:
            try:
                with open(f"../DataCollectionCode/Datasets/{d}/users/{id}/timeline.json") as f:
                    data = ujson.load(f)
                    f.close()
                if len(data) >= 10:
                    ids.append(id)
                else:
                    continue
            except:
                pass
        
        # Extract sample
        final = bots.loc[bots['id'].isin(ids)].sample(n=100, random_state=1234)
        final.to_csv(f"Data/{d}-GPT-sample.csv")

def create_raw_csv(d):
    
    tweets = []
    ids = []
    df = pd.read_csv(f"Data/{d}-GPT-sample.csv")
    for id in df['id']:
        with open(f"../DataCollectionCode/collected_data/{d}/users/{id}/timeline.json") as f:
            data = ujson.load(f)
            f.close()
        temp_tweets = [k['text'] for k in data]
        ids += [id for k in range(len(temp_tweets))]
        tweets += temp_tweets
    
    # Form pandas
    df_new = pd.DataFrame()
    df_new['raw_tweet'] = tweets
    df_new['id'] = ids
    df_new.to_csv(f"Data/{d}-raw-tweets.csv", index=False)

def request_chatgpt(d, set_no):

    i = int((20000/20) * (set_no - 1))
    j = int((20000/20) * (set_no))

    df = pd.read_csv(f"Data/{d}-raw-tweets.csv")
    client = OpenAI(
    # This is the default and can be omitted
        api_key="",
    )

    # Iterate over each file (top 100)
    responses = []
    for raw_tweet in df['raw_tweet'].iloc[i:j]:
        prompt_str = f"Please paraphrase and re-write the following tweet: {raw_tweet}."
        chat_completion = client.chat.completions.create(
            messages=[
                        {
                            "role": "user",
                            "content": prompt_str,
                        }
            ],
            model="gpt-3.5-turbo",
        )
        response_text = chat_completion.choices[0].message.content
        responses.append(response_text)
    
    # Create modified tweets csv
    df_new = pd.DataFrame()
    df_new['id'] = df['id'].iloc[i:j]
    df_new['raw_tweets'] =  df['raw_tweet'].iloc[i:j]
    df_new['rewritten_tweet'] = responses 
    df_new.to_csv(f"Data/{d}-sample-modified-tweets-{set_no}.csv", index=False)

if __name__ == "__main__":

    # Dataset
    d = sys.argv[1]
    # Set number
    set_no = int(sys.argv[2])
    # Run code as `python3 submit_prompts_chat.py gilani-2017 set_no`
    # Each set contains 1k tweets with a total of 20 sets
    request_chatgpt(d, set_no)
