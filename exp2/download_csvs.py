import pandas as pd
import wget
import sys

df = pd.read_csv(sys.argv[1])
for ix,row in df.iterrows():
    url = "https://manifesto-project.wzb.eu//tools/documents/2020-2/coded/{}_{}.csv".format(row[1],row[0])
    wget.download(url)