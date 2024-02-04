import support_functions as sf
import pandas as pd
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
from scipy.stats import bartlett


def transform_into_df(file, name):
    alg1_1 = sf.import_pickle_file(file)
    values = list(alg1_1.values())
    df = pd.DataFrame({name : values})
    return df

def plot(dfs):
    i = 1
    for df in dfs:
        df.hist()
        plt.title(f'Histogram of {i}')
        i+=1
        plt.show()

def transform_df_add_alg(dfs):
    i = 1
    for df in dfs:
        df['algoritme'] = i
        i+=1
    return dfs

def anova(df, value):
    df[value] = pd.to_numeric(df[value], errors='coerce')
    df['algoritme'] = pd.to_numeric(df['algoritme'], errors='coerce')
    df.dropna(subset=[value, 'algoritme'], inplace=True)
    anova_result = f_oneway(
        df[value][df['algoritme'] == 1],
        df[value][df['algoritme'] == 2],
        df[value][df['algoritme'] == 3],
        df[value][df['algoritme'] == 4],
    )
    print(f"One-way ANOVA Test Statistic: {anova_result.statistic}")
    print(f"P-value: {anova_result.pvalue}")
    alpha = 0.05
    if anova_result.pvalue < alpha:
        print("The differences between groups are statistically significant.")
    else:
        print("There is no significant difference between groups.")

    tukey_results = pairwise_tukeyhsd(df[value], df['algoritme'])
    print(tukey_results)

def barlett(df, value):
    df[value] = pd.to_numeric(df[value], errors='coerce')
    df.dropna(subset=[value, 'algoritme'], inplace=True)
    bartlett_result = bartlett(
        df[value][df['algoritme'] == 1],
        df[value][df['algoritme'] == 2],
        df[value][df['algoritme'] == 3],
        df[value][df['algoritme'] == 4],
)
    print(f"Bartlett's Test Statistic: {bartlett_result.statistic}")
    print(f"P-value: {bartlett_result.pvalue}")

#inladen van de profit df's
df1_1_profit = transform_into_df('data/data_algoritme1/performance_alg1_distance_heatmap1_1.pickle', 'profit')
df2_1_profit = transform_into_df('data/data_algoritme2/perf_alg2_veh1_profit.pickle', 'profit')
df3_1_profit = transform_into_df('data/data_algoritme3/performance_alg3_profit_heatmap1_1.pickle', 'profit')
df4_1_profit = transform_into_df('data/data_algoritme4/performance_alg4_profit_heatmap1_1.pickle', 'profit') #<--DEZE DUS VERANDEREN NAAR HET NIEUWE JUISTE BESTAND

df1_10_profit = transform_into_df('data/data_algoritme1/performance_alg1_profit_heatmap2_10.pickle', 'profit')
df2_10_profit = transform_into_df('data/data_algoritme2/perf_alg2_veh10_profit.pickle', 'profit')
df3_10_profit = transform_into_df('data/data_algoritme3/performance_alg3_profit_heatmap1_10.pickle', 'profit')
df4_10_profit = transform_into_df('data/data_algoritme4/performance_alg4_profit_heatmap1_10.pickle', 'profit') #<--DEZE DUS VERANDEREN NAAR HET NIEUWE JUISTE BESTAND

#inladen van de runtime df's

df1_1_runtime = transform_into_df('data/data_algoritme1/performance_alg1_runtime_heatmap2_1.pickle', 'runtime')
df2_1_runtime = transform_into_df('data/data_algoritme2/perf_alg2_veh1_runtime.pickle', 'runtime')
df3_1_runtime = transform_into_df('data/data_algoritme3/performance_alg3_runtime_heatmap1_1.pickle', 'runtime')
df4_1_runtime = transform_into_df('data/data_algoritme4/performance_alg4_runtime_heatmap1_1.pickle', 'runtime')  #<--DEZE DUS VERANDEREN NAAR HET NIEUWE JUISTE BESTAND

df1_10_runtime = transform_into_df('data/data_algoritme1/performance_alg1_runtime_heatmap2_10.pickle', 'runtime')
df2_10_runtime = transform_into_df('data/data_algoritme2/perf_alg2_veh10_runtime.pickle', 'runtime')
df3_10_runtime = transform_into_df('data/data_algoritme3/performance_alg3_runtime_heatmap1_10.pickle', 'runtime')
df4_10_runtime = transform_into_df('data/data_algoritme4/performance_alg4_runtime_heatmap1_10.pickle', 'runtime') #<--DEZE DUS VERANDEREN NAAR HET NIEUWE JUISTE BESTAND

dfs1_run = [df1_1_runtime, df2_1_runtime, df3_1_runtime, df4_1_runtime]
dfs1_run = transform_df_add_alg(dfs1_run)
df_run = pd.concat(dfs1_run, ignore_index=True)

dfs10_run = [df1_10_runtime, df2_10_runtime, df3_10_runtime, df4_10_runtime]
dfs10_run = transform_df_add_alg(dfs10_run)
df10_run = pd.concat(dfs10_run, ignore_index=True)

df_1_test2 = [df1_1_profit, df2_1_profit, df3_1_profit, df4_1_profit]
df_1_test = transform_df_add_alg(df_1_test2)
df_1 = pd.concat(df_1_test, ignore_index=True)

df_10_test2 = [df1_10_profit, df2_10_profit, df3_10_profit, df4_10_profit]
df_10_test = transform_df_add_alg(df_10_test2)
df_10 = pd.concat(df_10_test, ignore_index=True)

# barlett(df10_run, 'runtime') #barlett test voor 10 vehicles -runtime
# barlett(df_run, 'runtime') # barlett voror 1 vehicle -runtime
print("barlett 10")
# barlett(df_1, 'profit')# barlett voor 1 vehicle -profit
barlett(df_10, 'profit')#barlett test voor 10 vehicles -profit


# anova(df10_run, 'runtime') #anova test voor 10 vehicles -runtime
# anova(df_run, 'runtime') # anova voror 1 vehicle -runtime
print("anova 10")
#anova(df_1, 'profit')# anova voor 1 vehicle -profit
anova(df_10, 'profit')#anova test voor 10 vehicles -profit
