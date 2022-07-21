from glob import glob
import pandas as pd
import sys
import pdb

lang = sys.argv[1]

data_dir = './data/'
data_dir += lang
output_dir = './data/merged/{}.csv'.format(lang)


df_all = None
for fname in glob('{}/*.csv'.format(data_dir)):
    print("########"+fname+"###########")
    meta = fname.split('/')[-1]
    party, year_month = meta.replace('.csv', '').split('_')
    year = year_month[:4]
    month = year_month[4:]
    df = pd.read_csv(fname)
    df['Year'] = len(df) * [year]
    df['Party'] = len(df)  * [party]
    df['month'] = len(df) * [month]
    df['cmp_label'] = df['cmp_code'].astype(str).str.split('.').str[0]
    if df_all is None:
        df_all = df
    else:
        df_all = pd.concat([df_all,df])

df_all.to_csv(output_dir,index=False)





