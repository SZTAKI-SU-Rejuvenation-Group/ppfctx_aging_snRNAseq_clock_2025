import scanpy as sc
import numpy as np
import pandas as pd
import h5py
filename = "ppfctx_snRNAseq_processed_data_2024_05_16.h5seurat"

import warnings
warnings.filterwarnings("ignore")

cell_type_name = {1:"Oligodendrocytes", 2:"Astrocytes", 3:"Microglia", 4:"T_Cells", 5:"OPCs",
                 6:"Excitatory_Neurons", 7:"Inhibitory_Neurons", 8:"Endothelial_Cells",
                 9:"Fibroblast_like_Cells", 10:"Unknown", 11:"Unknown_Neurons"}

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
        
cols_to_keep = list(df.columns[:-21])

print("Start")
### pseudobulk
cols_to_keep1 = cols_to_keep.copy()
cols_to_keep1+=["donor_id","cell_type"]
df_main = df[cols_to_keep1]

cell_types = [1,2,3,5,6,7]

df_avg = df_main[df_main["cell_type"].isin(cell_types)]
df_avg = df_avg.groupby("donor_id").mean()
del df_avg["cell_type"]

impute_pseudobulk = pd.DataFrame(df_avg.mean()).T
impute_pseudobulk.columns = list(df_avg.columns)
impute_pseudobulk.to_csv("imputation_data/pseudobulk.csv", index=False)
print("Pseudobulk done")

### pseudobulk bootstrapped 500
cell_types = [1,2,3,5,6,7]

df2 = df[df["cell_type"].isin(cell_types)]

df_avg = pd.DataFrame()
donors = df2["donor_id"].unique()
for donor in donors:
    df_aux = df2[df2["donor_id"]==donor]
    if len(df_aux)>500:
        for j in range(100):
            df_sample = df_aux.sample(n=500, axis=0, ignore_index=True)
            df_sample_main = df_sample[cols_to_keep].mean()
            df_sample_final = pd.DataFrame(df_sample_main).T
            df_avg = pd.concat([df_avg, df_sample_final])
    else:
        for j in range(100):
            df_sample_main = df_aux[cols_to_keep].mean()
            df_sample_final = pd.DataFrame(df_sample_main).T
            df_avg = pd.concat([df_avg, df_sample_final])
impute_pseudobulk = pd.DataFrame(df_avg.mean()).T
impute_pseudobulk.columns = list(df_avg.columns)
impute_pseudobulk.to_csv("imputation_data/pseudobulk_bootstrapped_500.csv", index=False)
print("Pseudobulk bootstrapped 500 done")

### pseudobulk bootstrapped 100
cell_types = [1,2,3,5,6,7]

df2 = df[df["cell_type"].isin(cell_types)]

df_avg = pd.DataFrame()
donors = df2["donor_id"].unique()
for donor in donors:
    df_aux = df2[df2["donor_id"]==donor]
    if len(df_aux)>100:
        for j in range(100):
            df_sample = df_aux.sample(n=100, axis=0, ignore_index=True)
            df_sample_main = df_sample[cols_to_keep].mean()
            df_sample_final = pd.DataFrame(df_sample_main).T
            df_avg = pd.concat([df_avg, df_sample_final])
    else:
        for j in range(100):
            df_sample_main = df_aux[cols_to_keep].mean()
            df_sample_final = pd.DataFrame(df_sample_main).T
            df_avg = pd.concat([df_avg, df_sample_final])
impute_pseudobulk = pd.DataFrame(df_avg.mean()).T
impute_pseudobulk.columns = list(df_avg.columns)
impute_pseudobulk.to_csv("imputation_data/pseudobulk_bootstrapped_100.csv", index=False)
print("Pseudobulk bootstrapped 100 done")

### glia pseudobulk
cols_to_keep2 = cols_to_keep.copy()
cols_to_keep2+=["cell_type", "donor_id"]
df_main = df[cols_to_keep2]
df_div = df_main[df_main["cell_type"].isin([1,2,5])]

df_avg = df_div.groupby("donor_id").mean()
del df_avg["cell_type"]

impute_pseudobulk = pd.DataFrame(df_avg.mean()).T
impute_pseudobulk.columns = list(df_avg.columns)
impute_pseudobulk.to_csv("imputation_data/pseudobulk_glia.csv", index=False)
print("Glia done")

### neuron pseudobulk
cols_to_keep2 = cols_to_keep.copy()
cols_to_keep2+=["cell_type", "donor_id"]
df_main = df[cols_to_keep2]
df_nondiv = df_main[df_main["cell_type"].isin([6,7])]

df_avg = df_nondiv.groupby("donor_id").mean()
del df_avg["cell_type"]

impute_pseudobulk = pd.DataFrame(df_avg.mean()).T
impute_pseudobulk.columns = list(df_avg.columns)
impute_pseudobulk.to_csv("imputation_data/pseudobulk_neuron.csv", index=False)
print("Neuron done")

### glia pseudobulk bootstrapped 500
df_avg = pd.DataFrame()
donors = df_div["donor_id"].unique()
for donor in donors:
    df_aux = df_div[df_div["donor_id"]==donor]
    if len(df_aux)>500:
        for j in range(100):
            df_sample = df_aux.sample(n=500, axis=0, ignore_index=True)
            df_sample_main = df_sample[cols_to_keep].mean()
            df_sample_final = pd.DataFrame(df_sample_main).T
            df_avg = pd.concat([df_avg, df_sample_final])
    else:
        for j in range(100):
            df_sample_main = df_aux[cols_to_keep].mean()
            df_sample_final = pd.DataFrame(df_sample_main).T
            df_avg = pd.concat([df_avg, df_sample_final])
impute_pseudobulk = pd.DataFrame(df_avg.mean()).T
impute_pseudobulk.columns = list(df_avg.columns)
impute_pseudobulk.to_csv("imputation_data/pseudobulk_glia_bootstrapped_500.csv", index=False)
print("Glia bootstrapped 500 done")

### glia pseudobulk bootstrapped 100
df_avg = pd.DataFrame()
donors = df_div["donor_id"].unique()
for donor in donors:
    df_aux = df_div[df_div["donor_id"]==donor]
    if len(df_aux)>100:
        for j in range(100):
            df_sample = df_aux.sample(n=100, axis=0, ignore_index=True)
            df_sample_main = df_sample[cols_to_keep].mean()
            df_sample_final = pd.DataFrame(df_sample_main).T
            df_avg = pd.concat([df_avg, df_sample_final])
    else:
        for j in range(100):
            df_sample_main = df_aux[cols_to_keep].mean()
            df_sample_final = pd.DataFrame(df_sample_main).T
            df_avg = pd.concat([df_avg, df_sample_final])
impute_pseudobulk = pd.DataFrame(df_avg.mean()).T
impute_pseudobulk.columns = list(df_avg.columns)
impute_pseudobulk.to_csv("imputation_data/pseudobulk_glia_bootstrapped_100.csv", index=False)
print("Glia bootstrapped 100 done")

### neuron pseudobulk bootstrapped 500
df_avg = pd.DataFrame()
donors = df_nondiv["donor_id"].unique()
for donor in donors:
    df_aux = df_nondiv[df_nondiv["donor_id"]==donor]
    if len(df_aux)>500:
        for j in range(100):
            df_sample = df_aux.sample(n=500, axis=0, ignore_index=True)
            df_sample_main = df_sample[cols_to_keep].mean()
            df_sample_final = pd.DataFrame(df_sample_main).T
            df_avg = pd.concat([df_avg, df_sample_final])
    else:
        for j in range(100):
            df_sample_main = df_aux[cols_to_keep].mean()
            df_sample_final = pd.DataFrame(df_sample_main).T
            df_avg = pd.concat([df_avg, df_sample_final])
impute_pseudobulk = pd.DataFrame(df_avg.mean()).T
impute_pseudobulk.columns = list(df_avg.columns)
impute_pseudobulk.to_csv("imputation_data/pseudobulk_neuron_bootstrapped_500.csv", index=False)
print("Neuron bootstrapped 500 done")

### neuron pseudobulk bootstrapped 100
df_avg = pd.DataFrame()
donors = df_nondiv["donor_id"].unique()
for donor in donors:
    df_aux = df_nondiv[df_nondiv["donor_id"]==donor]
    if len(df_aux)>100:
        for j in range(100):
            df_sample = df_aux.sample(n=100, axis=0, ignore_index=True)
            df_sample_main = df_sample[cols_to_keep].mean()
            df_sample_final = pd.DataFrame(df_sample_main).T
            df_avg = pd.concat([df_avg, df_sample_final])
    else:
        for j in range(100):
            df_sample_main = df_aux[cols_to_keep].mean()
            df_sample_final = pd.DataFrame(df_sample_main).T
            df_avg = pd.concat([df_avg, df_sample_final])
impute_pseudobulk = pd.DataFrame(df_avg.mean()).T
impute_pseudobulk.columns = list(df_avg.columns)
impute_pseudobulk.to_csv("imputation_data/pseudobulk_neuron_bootstrapped_100.csv", index=False)
print("Neuron bootstrapped 100 done")

### cell type specific pseudobulk
cols_to_keep2 = cols_to_keep.copy()
cols_to_keep2+=["donor_id", "cell_type"]
df_main = df[cols_to_keep2]
cell_types = [1,2,3,5,6,7]
for cell_type in cell_types:
    df_current = df_main[df_main["cell_type"]==cell_type]
    df_current = df_current.groupby("donor_id").mean()
    del df_current["cell_type"]
    impute_pseudobulk = pd.DataFrame(df_current.mean()).T
    impute_pseudobulk.columns = list(df_current.columns)
    impute_pseudobulk.to_csv("imputation_data/pseudobulk_celltype_"+cell_type_name[cell_type]+".csv", index=False)
print("Ct pseudobulk done")

### cell type specific pseudobulk bootstrapped
num_cells = {1:200, 2:50, 3:50, 4:3, 5:50, 6:100, 7:100, 8:2, 9:5, 10:5, 11:3}
cell_types = [1,2,3,5,6,7]
for cell_type in cell_types:
    df_avg = pd.DataFrame()
    df_current1 = df[df["cell_type"]==cell_type]
    donors = df_current1["donor_id"].unique()
    for donor in donors:
        df_aux = df_current1[df_current1["donor_id"]==donor]
        if len(df_aux)>num_cells[cell_type]:
            for j in range(100):
                df_sample = df_aux.sample(n=num_cells[cell_type], axis=0, ignore_index=True)
                df_sample_main = df_sample[cols_to_keep].mean()
                df_sample_final = pd.DataFrame(df_sample_main).T
                df_avg = pd.concat([df_avg, df_sample_final])
        else:
            for j in range(100):
                df_sample_main = df_aux[cols_to_keep].mean()
                df_sample_final = pd.DataFrame(df_sample_main).T
                df_avg = pd.concat([df_avg, df_sample_final])
    impute_pseudobulk = pd.DataFrame(df_avg.mean()).T
    impute_pseudobulk.columns = list(df_avg.columns)
    impute_pseudobulk.to_csv("imputation_data/pseudobulk_celltype_bootstrapped_"+cell_type_name[cell_type]+".csv", index=False)
print("Ct pseudobulk bootstrapped done")

### single cell
cols_to_keep2 = cols_to_keep.copy()
cols_to_keep2+=["cell_type","donor_id"]
df_main = df[cols_to_keep2]
cell_types = [1,2,3,5,6,7]
for cell_type in cell_types:
    df_current = df_main[df_main["cell_type"]==cell_type]
    del df_current["cell_type"]
    del df_current["donor_id"]
    impute_sc = pd.DataFrame(df_current.mean()).T
    impute_sc.columns = list(impute_sc.columns)
    impute_sc.to_csv("imputation_data/sc_"+cell_type_name[cell_type]+".csv", index=False)
print("Sc done")