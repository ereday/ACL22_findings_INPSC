# take one of the merged manifesto file and convert it to desired file format
import pandas as pd
import sys
percentage=sys.argv[1]
train_set_ratio = int(percentage)/100
folder_name = "train_dev_test_"+percentage

for cname in ['turkey','spain','italy','hungary','germany','france','finland','united_kingdom']:
    fname = '../data/merged/{}.csv'.format(cname)
    foutname = '../data/{}/{}/{}.csv'.format(folder_name,cname,cname)
    foutname_trn = '../data/{}/{}/train.csv'.format(folder_name,cname)
    foutname_dev = '../data/{}/{}/dev.csv'.format(folder_name,cname)
    foutname_tst = '../data/{}/{}/test.csv'.format(folder_name,cname)
    df = pd.read_csv(fname,sep=',')
    df = df.loc[df['cmp_label']!='H'].reset_index(drop=True)
    # In hungarian dataset, there are very few instances with 3053 label which is not part of the codebook.
    if cname == 'hungary':
            df = df.loc[df['cmp_label'] != '3053'].reset_index(drop=True)

    df.loc[df.cmp_label=='0','cmp_label']='000'
    df = df[df['cmp_label'].notna()].reset_index(drop=True)
    df = df[df['text'].notna()].reset_index(drop=True)
    df = df.loc[df['cmp_label']!='000'].reset_index(drop=True)


    labels = df['cmp_label'].unique().tolist()
    labels = [int(label) for label in labels]
    majors = []
    for l in labels:
        if l == 0:
            continue
        majors.append(l - (l%100))

    major_classes = list(set(majors))
    all_classes = sorted(list(set(labels+major_classes)))

    data = []
    for ix,row in df.iterrows():
        text  = row.text
        minor_label = int(row.cmp_label)
        major_label = minor_label - (minor_label % 100)
        label_set = [1 if c == minor_label or c == major_label else 0 for c in all_classes ]
        new_row = [text]+label_set
        data.append(new_row)
    df2 = pd.DataFrame(data,columns=['input']+all_classes)
    train = df2.sample(frac=0.8, random_state=1234)  # random state is a seed value
    test = df2.drop(train.index)

    train = train.sample(frac=train_set_ratio,random_state = 1234)

    train_new = train.sample(frac=0.8, random_state=1234)
    dev = train.drop(train_new.index)
    df2.to_csv(foutname,index=None)
    #train.to_csv(foutname_trn, index=None)
    train_new.to_csv(foutname_trn,index=None)
    dev.to_csv(foutname_dev,index=None)
    test.to_csv(foutname_tst, index=None)