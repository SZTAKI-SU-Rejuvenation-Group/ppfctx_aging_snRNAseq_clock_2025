import scanpy as sc
import numpy as np
import pandas as pd
import h5py

### 
###
### reading expression data
###
###

filename = "ppfctx_snRNAseq_processed_data_2024_05_16.h5seurat"

with h5py.File(filename, "r") as f:
    cols = np.array(f["assays"]["RNA"]['features'])
    indices = np.array(f["cell.names"])
    age_list = np.array(f["meta.data"]["age"])
    donor_list = np.array(f["meta.data"]["sample_name"])
    
    spcd=np.array(f["assays"]["RNA"]["data"]["data"])
    spcind=np.array(f["assays"]["RNA"]["data"]["indices"])
    spcptr=np.array(f["assays"]["RNA"]["data"]["indptr"])
    sufnimtr=np.zeros(shape=(73941,36601),dtype=np.float32)
    for i in range(73941):
        indmsk=spcind[spcptr[i]:spcptr[i+1]]
        data=spcd[spcptr[i]:spcptr[i+1]]
        sufnimtr[i][indmsk]=data
    df=pd.DataFrame(sufnimtr,index=indices,columns=cols)
    df["age"] = age_list
    df["donor_id"] = donor_list
    
with h5py.File(filename, "r") as f:
    for info in np.array(f["meta.data"]):
        if len(np.array(f["meta.data"][info]))==73941:
            df[info] = np.array(f["meta.data"][info])
            
with h5py.File(filename, "r") as f:
    df["cell_type"] = np.array(f["meta.data"]["cell_type"]["values"])
    
with h5py.File(filename, "r") as f:
    df["cluster"] = np.array(f["meta.data"]["cluster_ann"]["values"])
    
columns = df.columns
columns2 = []
for col in columns:
    if type(col)==bytes:
        columns2.append(col.decode("utf8"))
    else:
        columns2.append(col)
        
df.columns = columns2

index = df.index
index2 = []
for ind in index:
    if type(ind)==bytes:
        index2.append(ind.decode("utf8"))
    else:
        index2.append(ind)
        
df.index = index2
    
for col in df.columns:
    if df[col].dtype=="O":
        new_col = df[col].str.decode("utf8")
        df[col]=new_col
        

genes = set((df.columns[:-21]))

sex_related_genes = pd.read_csv("X_Y_MT_gene_list_hg38.txt")
sex_related_genes["Gene name"].fillna(sex_related_genes["Gene stable ID"],inplace=True)
genes_sexmt = set(sex_related_genes["Gene name"])

cols_to_keep_genes = list(genes-genes_sexmt)
cols_to_keep = list(genes-genes_sexmt)

cols_to_keep+=["age", "cell_type", "donor_id", "seq_batch"]
df_main = df[cols_to_keep]
df_main["seq_batch"] = df_main["seq_batch"].str.strip("batch").astype(int)

df_div = df_main[df_main["cell_type"].isin([1,2,5])]


###
###
### training glia-level simple pseudobulk aging clocks
###
###

from sklearn.model_selection import cross_validate, KFold
from glmnet import ElasticNet as glmelastic
from sklearn.metrics import mean_absolute_error
from scipy import stats

df_avg = df_div.groupby("donor_id").mean()
del df_avg["cell_type"]
df_avg = df_avg.reset_index()

alphas=[0,0.5,1]
for alpha in alphas:
    model = glmelastic(alpha=alpha, random_state=42)
    cv_results = cross_validate(model, df_avg[df_avg.columns[1:-2]], df_avg["age"],
                                scoring = ["neg_mean_absolute_error"], 
                                cv = KFold(n_splits=5, shuffle=True, random_state=0), return_estimator = True, return_indices = True)
    df_final = pd.DataFrame()
    df_models = pd.DataFrame()
    for i in range(5):
        coeff = cv_results["estimator"][i].coef_
        intercept = cv_results["estimator"][i].intercept_
        coefficients = np.append(coeff, intercept)
        col_names = np.append(list(df_avg.columns[1:-2]), "intercept")
        model_df = pd.DataFrame(coefficients.reshape(-1, len(coefficients)),columns=col_names)
        df_models = pd.concat([df_models, model_df], ignore_index=True)
        
        df_current = df_avg.iloc[cv_results["indices"]["test"][i]][["age","seq_batch", "donor_id"]]
        predicted = cv_results["estimator"][i].predict(df_avg.iloc[cv_results["indices"]["test"][i]][df_avg.columns[1:-2]])
        df_current["predicted_age"] = list(predicted)
        df_final = pd.concat([df_final, df_current])
    
    df_models.to_csv("clocks/pseudobulk_glia_"+str(alpha)+".csv", index=False)
    df_final.to_csv("results/pseudobulk_glia_"+str(alpha)+".csv", index=False)
    
###
###
### training glia-level bootstrapped pseudobulk aging clocks (with 500 cells sampled)
###
###

from sklearn.model_selection import GroupKFold

df_avg = pd.DataFrame()
donors = df_div["donor_id"].unique()
for donor in donors:
    df_aux = df_div[df_div["donor_id"]==donor]
    if len(df_aux)>500:
        for j in range(100):
            df_sample = df_aux.sample(n=500, axis=0, ignore_index=True)
            df_sample_main = df_sample[cols_to_keep_genes].mean()
            df_sample_meta = df_aux[["donor_id", "age", "seq_batch"]].iloc[0]
            df_sample_final = pd.DataFrame(pd.concat([df_sample_main, df_sample_meta])).T
            df_avg = pd.concat([df_avg, df_sample_final])
    else:
        for j in range(100):
            df_sample_main = df_aux[cols_to_keep_genes].mean()
            df_sample_meta = df_aux[["donor_id", "age", "seq_batch"]].iloc[0]
            df_sample_final = pd.DataFrame(pd.concat([df_sample_main, df_sample_meta])).T
            df_avg = pd.concat([df_avg, df_sample_final])
df_avg.index = range(len(df_avg))

alphas=[0,0.5,1]
df_res = pd.DataFrame()
for alpha in alphas:
    model = glmelastic(alpha=alpha, random_state=42)
    cv_results = cross_validate(model, df_avg[df_avg.columns[:-3]], df_avg["age"], groups = df_avg["donor_id"],
                                scoring = ["neg_mean_absolute_error"], 
                                cv = GroupKFold(n_splits=5), 
                                return_estimator = True, return_indices = True)
    df_final = pd.DataFrame()
    df_models = pd.DataFrame()
    for i in range(5):
        coeff = cv_results["estimator"][i].coef_
        intercept = cv_results["estimator"][i].intercept_
        coefficients = np.append(coeff, intercept)
        col_names = np.append(list(df_avg.columns[:-3]), "intercept")
        model_df = pd.DataFrame(coefficients.reshape(-1, len(coefficients)),columns=col_names)
        df_models = pd.concat([df_models, model_df], ignore_index=True)
        
        df_current = df_avg.iloc[cv_results["indices"]["test"][i]][["age", "donor_id", "seq_batch"]]
        predicted = cv_results["estimator"][i].predict(df_avg.iloc[cv_results["indices"]["test"][i]][df_avg.columns[:-3]])
        df_current["predicted_age"] = list(predicted)
        df_final = pd.concat([df_final, df_current])
    
    df_models.to_csv("clocks/pseudobulk_glia_500_bootstrapped_"+str(alpha)+".csv", index=False)
    df_final.to_csv("results/predictions_pseudobulk_glia_500_bootstrapped_"+str(alpha)+"_.csv")
    
###
###
### training glia-level bootstrapped pseudobulk aging clocks (with 100 cells sampled)
###
###

df_avg = pd.DataFrame()
donors = df_div["donor_id"].unique()
for donor in donors:
    df_aux = df_div[df_div["donor_id"]==donor]
    if len(df_aux)>100:
        for j in range(100):
            df_sample = df_aux.sample(n=100, axis=0, ignore_index=True)
            df_sample_main = df_sample[cols_to_keep_genes].mean()
            df_sample_meta = df_aux[["donor_id", "age", "seq_batch"]].iloc[0]
            df_sample_final = pd.DataFrame(pd.concat([df_sample_main, df_sample_meta])).T
            df_avg = pd.concat([df_avg, df_sample_final])
    else:
        for j in range(100):
            df_sample_main = df_aux[cols_to_keep_genes].mean()
            df_sample_meta = df_aux[["donor_id", "age", "seq_batch"]].iloc[0]
            df_sample_final = pd.DataFrame(pd.concat([df_sample_main, df_sample_meta])).T
            df_avg = pd.concat([df_avg, df_sample_final])
df_avg.index = range(len(df_avg))

alphas=[0,0.5,1]
df_res = pd.DataFrame()
for alpha in alphas:
    model = glmelastic(alpha=alpha, random_state=42)
    cv_results = cross_validate(model, df_avg[df_avg.columns[:-3]], df_avg["age"], groups = df_avg["donor_id"],
                                scoring = ["neg_mean_absolute_error"], 
                                cv = GroupKFold(n_splits=5), 
                                return_estimator = True, return_indices = True)
    df_final = pd.DataFrame()
    df_models = pd.DataFrame()
    for i in range(5):
        coeff = cv_results["estimator"][i].coef_
        intercept = cv_results["estimator"][i].intercept_
        coefficients = np.append(coeff, intercept)
        col_names = np.append(list(df_avg.columns[:-3]), "intercept")
        model_df = pd.DataFrame(coefficients.reshape(-1, len(coefficients)),columns=col_names)
        df_models = pd.concat([df_models, model_df], ignore_index=True)
        
        df_current = df_avg.iloc[cv_results["indices"]["test"][i]][["age", "donor_id", "seq_batch"]]
        predicted = cv_results["estimator"][i].predict(df_avg.iloc[cv_results["indices"]["test"][i]][df_avg.columns[:-3]])
        df_current["predicted_age"] = list(predicted)
        df_final = pd.concat([df_final, df_current])
    
    df_models.to_csv("clocks/pseudobulk_glia_100_bootstrapped_"+str(alpha)+".csv", index=False)
    df_final.to_csv("results/predictions_pseudobulk_glia_100_bootstrapped_"+str(alpha)+"_.csv")
    