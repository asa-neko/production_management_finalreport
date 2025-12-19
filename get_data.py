def get_data(url):
    import requests
    import pandas as pd

    df = pd.read_excel(url)
    return df

df = get_data('http://misc.0093.tv/misc/kadai.xlsx')
df.to_excel('data/kadai.xlsx', index=False)
