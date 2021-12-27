import pandas as pd
import json

def json2df(path):
    data = json.load(open(path,"r"))
    df = pd.DataFrame.from_dict(data)
    return df

print(json2df("test_100k.json"))
