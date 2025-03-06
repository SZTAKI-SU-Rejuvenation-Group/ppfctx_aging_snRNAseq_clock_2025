print("It works!")
import pandas as pd
import scanpy as sc
import numpy as np
import os
import json
import anndata as ad
import scipy
import h5py

### reading the data that we want to apply the clock to

adata = ad.read('./external_data/frohlich_control_lognorm_2024-12-19.h5ad', backed='r')

with h5py.File("external_data/frohlich_control_lognorm_2024-12-19.h5seurat", "r") as f:
    ct_list = list(f["meta.data"]["major_celltypes_relabel"]["levels"])
    ct_list =[x.decode('utf-8') for x in ct_list]
    ct_map = {}
    for i in range(len(ct_list)):
        ct_map[i] = ct_list[i]

with h5py.File("external_data/frohlich_control_lognorm_2024-12-19.h5seurat", "r") as f:
    donor_list = list(f["meta.data"]["Donor"]["levels"])
    donor_list =[x.decode('utf-8') for x in donor_list]
    donor_map = {}
    for i in range(len(donor_list)):
        donor_map[i] = donor_list[i]

adata.obs['major_celltypes_relabel'] = adata.obs['major_celltypes_relabel'].map(ct_map) 
adata.obs['Donor'] = adata.obs['Donor'].map(donor_map) 

# number of cells to be sampled of each cell type for the bootstrapped pseudobulk clocks
num_cells = {"Oligodendrocytes":200, "Astrocytes":50, "Microglia":50, "OPCs":50, "Excitatory_Neurons":100, "Inhibitory_Neurons":100}

# mapping cell type names to match the notations in the training dataset on which the clocks were trained on
ct_map = {
    "Inhibitory_Neurons":"Inhibitory_Neurons",
    "Oligodendrocytes":"Oligodendrocytes",
    "Astrocytes":"Astrocytes",
    "Excitatory_Neurons":"Excitatory_Neurons",
    "OPCs":"OPCs",
    "Microglia":"Microglia"
}

ct_map_clock = {
    "Inhibitory_Neurons":"7",
    "Oligodendrocytes":"1",
    "Astrocytes":"2",
    "Excitatory_Neurons":"6",
    "OPCs":"5",
    "Microglia":"3"
}

### this function reads data from a h5ad file only for the given cell type into a dataframe -- this is a memory saving way to read the needed data, because in this way we don't load the entire h5ad file into memory
def get_df(data, cell_type, sample_type):
    ###########
        # data: a loaded anndata file, but it does not have to be loaded into memory, loading with "backed='r'" option is enough
        # cell_type: cell type in a string format, that is to be loaded into memory; it has to be character-wisely the same as the one in the "major_celltypes_relabel" column of the data.obs dataframe
        # sample_type: single-cell, simple or bootstrapped pseudobulk version is called (sc, ct, ct_sample)
    ###########
    
    k=data[data.obs["major_celltypes_relabel"]==cell_type]
    sh=(k.obs.shape[0],k.var.shape[0])
    cols=k.var_names
    indexes=k.obs_names
    
    spcd=k.X.data
    spcind=k.X.indices
    spcptr=k.X.indptr

    sufnimtr=np.zeros(shape=sh,dtype=np.float32)
    for i in range(sh[0]):
        indmsk=spcind[spcptr[i]:spcptr[i+1]]
        data=spcd[spcptr[i]:spcptr[i+1]]
        sufnimtr[i][indmsk]=data
    df=pd.DataFrame(sufnimtr,index=indexes,columns=cols)
    
    ### adding age and donor id info to the dataframe
    age_list = k.obs["Age"]
    donor_list = k.obs["Donor"]
    cell_name_list = k.obs.index
    df["age"] = age_list
    df["donor_id"] = donor_list
    cols_to_keep = df.columns[:-2]
    
    if sample_type=="sc":
        df["cell_name"] = cell_name_list
        df.set_index("cell_name", inplace=True)
    elif sample_type=="ct":
        df = df.groupby("donor_id").mean()
        df = df.reset_index()
    elif sample_type=="ct_sample":
        df_avg = pd.DataFrame()
        donors = df["donor_id"].unique()
        for donor in donors:
            df_aux = df[df["donor_id"]==donor]
            if len(df_aux)>num_cells[cell_type]:
                for j in range(100):
                    df_sample = df_aux.sample(n=num_cells[cell_type], axis=0, ignore_index=True)
                    df_sample_main = df_sample[cols_to_keep].mean()
                    df_sample_meta = df_aux[["donor_id", "age"]].iloc[0]
                    df_sample_final = pd.DataFrame(pd.concat([df_sample_main, df_sample_meta])).T
                    df_avg = pd.concat([df_avg, df_sample_final])
            else:
                for j in range(100):
                    df_sample_main = df_aux[cols_to_keep].mean()
                    df_sample_meta = df_aux[["donor_id", "age"]].iloc[0]
                    df_sample_final = pd.DataFrame(pd.concat([df_sample_main, df_sample_meta])).T
                    df_avg = pd.concat([df_avg, df_sample_final])
        df_avg.index = range(len(df_avg))
        df = df_avg
    
    return df


### applying one clock to data from one cell type (or any data fed to the function)
### it is assumed that the name of the genes are the same in the training and application dataset, thus they can be mapped onto each other directly
### for more details see the comments inside the function
def apply_clock_celltype(data, impute_data, model, sample_type):
    ###########
        # data: dataframe containing the normalized expression counts and age and donor info --> the output of the get_df function
        # impute_data: dataframe containing imputation values for genes in the training dataset --> it is used to impute missing values (expression value of genes that are not present in the application dataset)
        # model: clock to be applied, in a dataframe format containing the coefficients and the intercept value
        # sample_type: single-cell, simple or bootstrapped pseudobulk version is called (sc, ct, ct_sample)
    ###########
    
    predictions = []
    ages = []
    donors = []
    cell_names = []
    
    ### taking the genes that are present in both the set of clock features and the application dataset
    genes_in_model = set(model.columns)&set(data.columns)
            
    ### keeping only the common genes in the application dataset + donor and age info
    cols_to_keep = list(genes_in_model)+["donor_id", "age"]
    df_celltype_age = data[cols_to_keep]
    
    ### coefficients and intercept of the clock
    model_coeff = model[model.columns[:-1]]
    model_intercept = model["intercept"]
    
    ### getting the set of genes that is present in the model, but is missing from the application dataset, and getting imputation values for these genes
    genes_rem = list(set(model_coeff.columns)-set(genes_in_model))
    data_for_imputation = impute_data[genes_rem]
    
    ### applying the clock to each cell/sample, one-by-one
    for i in range(len(df_celltype_age)):
        if sample_type=="sc":
            cell_name = df_celltype_age.iloc[i].name
        ### missing value imputation
        row = pd.concat([df_celltype_age.iloc[i].to_frame().T.reset_index(drop=True), data_for_imputation], axis=1)
    
        donor = row["donor_id"]
        age = row["age"]
        row = row[list(set(row.columns)-set(["donor_id","age"]))]
        
        ### applying the clock --> dot product of the expression values and the model coefficients + intercept
        prediction = row.dot(model_coeff.T) + model_intercept
    
        predictions.append(prediction[0][0])
        ages.append(age[0])
        donors.append(donor[0])
        if sample_type=="sc":
            cell_names.append(cell_name)
    
    
    ### output: list of predictions, ages, donor ids, cell names --> each element of the lists represent one cell/sample
    return predictions, ages, donors, cell_names


### this is the main function
### the function "apply_clock_celltype" is applied with all 5 models of a given cell type specific clock (resulting from 5-fold cv)
def predict(df_base, cell_type, dataset_name, clock_file, imputation_file, sample_type):
    ###########
        # df_base: dataframe containing the normalized expression counts and age and donor info --> the output of the get_df function
        # cell_type: cell type in a string format
        # dataset_name: technical argument for the name of the resulting file
        # clock_file: name of the file containing the clock to be applied
        # imputation_file: name of the file containing the data for the imputation of missing values
        # sample_type: single-cell, simple or bootstrapped pseudobulk version is called (sc, ct, ct_sample)
    ###########
    
    ### reading imputation data and models
    impute_data = pd.read_csv("./imputation_data/"+imputation_file)
    models_celltype = pd.read_csv("./clocks/"+clock_file)
    
    df_final = pd.DataFrame()
    
    ### applying the 5 clocks one-by-one
    for j in range(len(models_celltype)):
        ### taking one clock from the dataframe of all 5 clocks
        current_model = models_celltype.iloc[j].to_frame().T.reset_index()
        current_model = current_model.drop("index",axis=1)
        
        ### taking only the non-zero coefficients from the clock
        m2 = (abs(current_model) > 0).any()
        a = m2.index[m2]
        model_to_use = current_model[list(a)]
        
        ### applying "apply_clock_celltype" with the current clock
        pred, ages, donors, cell_names = apply_clock_celltype(df_base, impute_data, model_to_use, sample_type)
        
        ### adding the predictions given by the current clock to the final dataframe, with donor, age and cell name info
        df = pd.DataFrame()
        df["donor"] = donors
        df["age"] = ages
        df["predicted_age"] = pred
        if sample_type=="sc":
            df["cell_name"] = cell_names
        df_final = pd.concat([df_final, df])
        
    ### writing the final datarframe into a csv file; this can be changed if needed
    df_final.to_csv("./application_of_clocks/"+dataset_name+"_"+ct_map[cell_type]+".csv", index=False)
    
    
    
    
### running the prediction process

### cell-type-specific single-cell
sample_type="sc"
print("sc")
i=1
for cell_type in common_celltypes:
    imputation_file = "sc_"+ct_map[cell_type]+".csv"
    df_base = get_df(adata, cell_type, sample_type)
    for param in [0.5]:
        clock_file = "sc_"+ct_map_clock[cell_type]+"_"+str(param)+".csv"
        predict(df_base, cell_type, "frohlich_sc_lognormed_"+str(param), clock_file, imputation_file, sample_type)
        print(str(i)+"/"+str(len(common_celltypes)*3))
        i+=1

### cell-type-specific simple pseudobulk
print("Start")
i=1
sample_type="ct"
print("ct")
for cell_type in common_celltypes:
    imputation_file = "pseudobulk_celltype_"+ct_map[cell_type]+".csv"
    df_base = get_df(adata, cell_type, sample_type)
    for param in [0.5]:
        clock_file = "pseudobulk_celltype_"+ct_map_clock[cell_type]+"_"+str(param)+".csv"
        predict(df_base, cell_type, "frohlich_celltype_pseudobulk_lognormed_"+str(param), clock_file, imputation_file, sample_type)
        print(str(i)+"/"+str(len(common_celltypes)*3))
        i+=1
    
### cell-type-specific bootstrapped pseudobulk
sample_type="ct_sample"
print("ct_sample")
i=1
for cell_type in common_celltypes:
    imputation_file = "pseudobulk_celltype_bootstrapped_"+ct_map[cell_type]+".csv"
    df_base = get_df(adata, cell_type, sample_type)
    for param in [0.5]:
        clock_file = "pseudobulk_celltype_bootstrapped_"+ct_map_clock[cell_type]+"_"+str(param)+".csv"
        predict(df_base, cell_type, "frohlich_celltype_pseudobulk_bootstrapped_lognormed_"+str(param), clock_file, imputation_file, sample_type)
        print(str(i)+"/"+str(len(common_celltypes)*3))
        i+=1
