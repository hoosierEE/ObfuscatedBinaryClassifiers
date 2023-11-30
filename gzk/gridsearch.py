import matplotlib.pyplot as plt, seaborn as sns
import numpy as np, pandas as pd

# numbers based on running
data = lambda: '''
samples neighbors acc
100     5         0.300
200     5         0.650
400     5         0.725
800     5         0.838
1600    5         0.881
2500    5         0.932
3200    5         0.950
5000    5         0.962
6400    5         0.966
100     10        0.200
200     10        0.600
400     10        0.650
800     10        0.838
1600    10        0.887
2500    10        0.932
3200    10        0.947
5000    10        0.966
6400    10        0.973
100     15        0.200
200     15        0.400
400     15        0.575
800     15        0.775
1600    15        0.850
2500    15        0.908
3200    15        0.956
5000    15        0.954
6400    15        0.967
'''

def main():
 d = np.array([[float(x) for x in line.split()] for line in data().splitlines()[2:]])
 return pd.DataFrame(d, columns='samples neighbors accuracy'.split())

def plot(df):
 plt.clf();plt.close('all')
 fig,ax = plt.subplots()
 p1 = sns.lineplot(data=df,x='samples',y='accuracy',markers=True,dashes=False,style='neighbors',hue='neighbors')
 palette = sns.color_palette('rocket_r')
 mx = dict(df.max())
 ax.text(mx['samples']*0.85,mx['accuracy']*0.9,f"max: {mx['accuracy']}")
 return ax

if __name__ == "__main__":
 ax = plot(main())
 plt.savefig('../../paper/images/knn-gridsearch.pdf', dpi=500, bbox_inches='tight')
