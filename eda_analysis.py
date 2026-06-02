
import pandas as pd
import matplotlib.pyplot as plt

def run_eda():
    df=pd.read_csv('data/EDA_Practice_Dataset.csv')

    df['Age']=df['Age'].fillna(df['Age'].median())
    df['Salary']=df['Salary'].fillna(df['Salary'].median())
    df['Purchase_Amount']=df['Purchase_Amount'].fillna(df['Purchase_Amount'].median())

    corr=df[['Age','Salary','Purchase_Amount']].corr()

    plt.figure(figsize=(5,4))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)),corr.columns,rotation=45)
    plt.yticks(range(len(corr.columns)),corr.columns)
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()

    plt.figure(figsize=(5,4))
    plt.scatter(df['Salary'],df['Purchase_Amount'])
    plt.xlabel('Salary')
    plt.ylabel('Purchase Amount')
    plt.tight_layout()
    plt.savefig('outputs/salary_vs_purchase.png')
    plt.close()

    city=df['City'].fillna('Unknown').value_counts().head(5)
    plt.figure(figsize=(5,4))
    city.plot(kind='bar')
    plt.tight_layout()
    plt.savefig('outputs/city_analysis.png')
    plt.close()

    with open('outputs/insights_report.txt','w') as f:
        f.write('Average Age: %.2f\n' % df['Age'].mean())
        f.write('Average Salary: %.2f\n' % df['Salary'].mean())
        f.write('Average Purchase: %.2f\n' % df['Purchase_Amount'].mean())

    print(df.describe())
