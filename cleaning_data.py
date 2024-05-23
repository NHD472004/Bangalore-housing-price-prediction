import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('./data/bengaluru_house_prices.csv',
                 usecols=['location', 'size', 'total_sqft', 'bath', 'balcony', 'price'])


# handle missing data
df[['bath', 'balcony']] = df[['bath', 'balcony']].fillna(0)
df = df.dropna()


# handle size format
df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))


# handle total_sqft
def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


def cvt_sqft_to_num(x):
    tokens = x.split(' - ')
    if len(tokens) == 2:
        return (float(tokens[0]) + float(tokens[1])) / 2
    try:
        return float(x)
    except:
        return None


df['total_sqft'] = df['total_sqft'].apply(cvt_sqft_to_num)
df = df[df['total_sqft'].notnull()]


# price per sqft
df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']


# handle location
df['location'] = df['location'].apply(lambda x: x.strip())
location_stats = df.groupby('location')['location'].agg('count').sort_values(ascending=False)
df['location'] = df['location'].apply(lambda x: 'other' if x in location_stats[location_stats <= 10] else x)


# handle outlier
df = df[~(df['total_sqft'] / df['bhk'] < 300)]

upper_limit = df['price_per_sqft'].mean() + df['price_per_sqft'].std()
lower_limit = df['price_per_sqft'].mean() - df['price_per_sqft'].std()
df = df[(df['price_per_sqft'] <= upper_limit) & (df['price_per_sqft'] > lower_limit)]


# check unreasonable
def plot_scatter_chart(dataframe, location):
    bhk2 = dataframe[(dataframe['location'] == location) & (dataframe['bhk'] == 2)]
    bhk3 = dataframe[(dataframe['location'] == location) & (dataframe['bhk'] == 3)]
    plt.scatter(x=bhk2['total_sqft'], y=bhk2['price'], color='blue', marker='*', label='2 bhk', s=50)
    plt.scatter(x=bhk3['total_sqft'], y=bhk3['price'], color='green', marker='+', label='3 bhk', s=50)
    plt.title(location)
    plt.show()


# print(df['location'])
# plot_scatter_chart(df, 'Rajaji Nagar')
print(df.head())
df.to_csv('./data/bhk.csv', index=False)