import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - Read the data
df = pd.read_csv("medical_examination.csv")

# 2 - Add 'overweight' column (BMI > 25 → 1 else 0)
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)

# 3 - Normalize cholesterol and gluc: 0 = good, 1 = bad
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5 - Melt the data into long format
    df_cat = pd.melt(df,
                     id_vars=['cardio'],
                     value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6 - Group and count the data
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()

    # 7 - Rename 'size' column to 'total' for plotting
    df_cat = df_cat.rename(columns={'size': 'total'})

    # 8 - Plot using seaborn catplot
    fig = sns.catplot(
        data=df_cat,
        kind='bar',
        x='variable',
        y='total',
        hue='value',
        col='cardio'
    ).fig

    # 9
    fig.savefig('catplot.png')
    return fig

# 10
def draw_heat_map():
    # 11 - Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12 - Calculate correlation matrix
    corr = df_heat.corr()

    # 13 - Create a mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Set up matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15 - Draw the heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)

    # 16
    fig.savefig('heatmap.png')
    return fig
