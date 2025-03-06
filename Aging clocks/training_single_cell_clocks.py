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

cols_to_keep = list(genes-genes_sexmt)

cols_to_keep+=["age", "cell_type", "donor_id", "index"]
df_main_base = df.reset_index()
df_main = df_main_base[cols_to_keep]

###
###
### training cell-type-specific single-cell aging clocks
###
###

from sklearn.model_selection import cross_validate, KFold
from glmnet import ElasticNet as glmelastic
from sklearn.metrics import mean_absolute_error
from scipy import stats

cell_type_name = {1:"Oligodendrocytes", 2:"Astrocytes", 3:"Microglia", 4:"T_Cells", 5:"OPCs",
                 6:"Excitatory_Neurons", 7:"Inhibitory_Neurons", 8:"Endothelial_Cells",
                 9:"Fibroblast_like_Cells", 10:"Unknown", 11:"Unknown_Neurons"}

from sklearn.model_selection import GroupKFold

cell_types = [1,2,3,5,6,7]
for cell_type in cell_types:
    print(cell_type)
    df_current = df_main[df_main["cell_type"]==cell_type]
    
    alphas=[0,0.5,1]
    for alpha in alphas:
        model = glmelastic(alpha=alpha, random_state=42)
        cv_results = cross_validate(model, df_current[df_current.columns[:-4]], df_current["age"], groups = df_current["donor_id"], 
                                scoring = ["neg_mean_absolute_error"], 
                                cv = GroupKFold(n_splits=5), return_estimator = True, return_indices = True)
        df_final = pd.DataFrame()
        df_models = pd.DataFrame()
        for i in range(5):
            coeff = cv_results["estimator"][i].coef_
            intercept = cv_results["estimator"][i].intercept_
            coefficients = np.append(coeff, intercept)
            col_names = np.append(list(df_current.columns[:-4]), "intercept")
            model_df = pd.DataFrame(coefficients.reshape(-1, len(coefficients)),columns=col_names)
            df_models = pd.concat([df_models, model_df], ignore_index=True)
            
            
            df_current2 = df_current.iloc[cv_results["indices"]["test"][i]][["age", "cell_type", "donor_id", "index"]]
            predicted = cv_results["estimator"][i].predict(df_current.iloc[cv_results["indices"]["test"][i]][df_current.columns[:-4]])
            df_current2["predicted_age"] = list(predicted)
            df_final = pd.concat([df_final, df_current2])
    
        df_models.to_csv("clocks/sc_"+str(cell_type)+"_"+str(alpha)+".csv", index=False)
        df_final.to_csv("results/predictions_sc_"+str(cell_type)+"_"+str(alpha)+".csv", index=False)