print("It works!")
import pandas as pd
import scanpy as sc
import numpy as np
import os
import json
import anndata as ad

### reading the data that we want to apply the clock to

adata = ad.read('./external_data/velmeshev_fc_adult.h5ad', backed='r')

# reading dictionary mapping gene ids to gene names
mapping_df = pd.read_csv("./mapping.csv")
mapping_dict = mapping_df.set_index('From').T.to_dict("records")
mapping_dict = mapping_dict[0]

# mapping cell type names to match the notations in the training dataset on which the clocks were trained on
ct_map = {
    "IN":"Inhibitory_Neurons",
    "OL":"Oligodendrocytes",
    "AST":"Astrocytes",
    "VASC":"Endothelial_Cells",
    "ExNeu":"Excitatory_Neurons",
    "OPC":"OPCs",
    "MG":"Microglia",
    "OUT":""
}

ct_map_clock = {
    "IN":"7",
    "OL":"1",
    "AST":"2",
    "VASC":"8",
    "ExNeu":"6",
    "OPC":"5",
    "MG":"3",
    "OUT":""
}

### this function reads data from a h5ad file only for the given cell type into a dataframe -- this is a memory saving way to read the needed data, because in this way we don't load the entire h5ad file into memory
def get_df(data, mapping, sample_type, pb_type):
    ###########
        # data: a loaded anndata file, but it does not have to be loaded into memory, loading with "backed='r'" option is enough
        # mapping: dictionary mapping gene ids to the gene names
        # sample_type: simple or bootstrapped version is called (pb, pb_sampled500, pb_sampled100)
        # pb_type: all-cells pseudobulk or glia/neuron-level pseudobulk version is called (whole, glia, neuron)
    ###########
    
    if pb_type == "whole":
        k=data.to_memory()
    elif pb_type == "glia":
        k=data[data.obs["Lineage"].isin(["OL","AST","OPC"])]
    elif pb_type == "neuron":
        k=data[data.obs["Lineage"].isin(["IN","ExNeu"])]
    
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
    age_list = k.obs["Age"].str.strip("yr").astype(int)
    donor_list = k.obs["donor_id"]
    #cell_name_list = k.obs["Individual"]
    df["age"] = age_list
    df["donor_id"] = donor_list
    cols_to_keep = df.columns[:-2]

    if sample_type=="pb":
        df = df.groupby("donor_id").mean()
        df = df.reset_index()
    elif sample_type=="pb_sampled500":
        df_avg = pd.DataFrame()
        donors = df["donor_id"].unique()
        for donor in donors:
            df_aux = df[df["donor_id"]==donor]
            if len(df_aux)>500:
                for j in range(100):
                    df_sample = df_aux.sample(n=500, axis=0, ignore_index=True)
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
    elif sample_type=="pb_sampled100":
        df_avg = pd.DataFrame()
        donors = df["donor_id"].unique()
        for donor in donors:
            df_aux = df[df["donor_id"]==donor]
            if len(df_aux)>100:
                for j in range(100):
                    df_sample = df_aux.sample(n=100, axis=0, ignore_index=True)
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
    ### rename columns (genes) to match the clock features
    df = df.rename(columns=mapping)
    
    return df


### applying one clock to a given type of data
### it is assumed that the name of the genes are the same in the training and application dataset, thus they can be mapped onto each other directly
### for more details see the comments inside the function
def apply_clock(data, impute_data, model, sample_type):
    ###########
        # data: dataframe containing the normalized expression counts and age and donor info --> the output of the get_df function
        # impute_data: dataframe containing imputation values for genes in the training dataset --> it is used to impute missing values (expression value of genes that are not present in the application dataset)
        # model: clock to be applied, in a dataframe format containing the coefficients and the intercept value
        # sample_type: simple or bootstrapped version is called (pb, pb_sampled500, pb_sampled100)
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
    
    ### applying the clock to each sample, one-by-one
    for i in range(len(df_celltype_age)):
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
    
    
    ### output: list of predictions, ages, donor ids, cell names --> each element of the lists represent one sample
    return predictions, ages, donors, cell_names


### this is the main function
### the function "apply_clock" is applied with all 5 models of a given clock (resulting from 5-fold cv)
def predict(df_base, dataset_name, mapping, clock_file, imputation_file, sample_type):
    ###########
        # df_base: dataframe containing the normalized expression counts and age and donor info --> the output of the get_df function
        # dataset_name: technical argument for the name of the resulting file
        # mapping: dictionary mapping gene ids to gene names
        # clock_file: name of the file containing the clock to be applied
        # imputation_file: name of the file containing the data for the imputation of missing values
        # sample_type: simple or bootstrapped version is called (pb, pb_sampled500, pb_sampled100)
    ###########
    
    ### reading imputation data and models
    impute_data = pd.read_csv("./imputation_data/v3/"+imputation_file)
    models_celltype = pd.read_csv("./clocks/v3/"+clock_file)
    
    ### reading the expression data (application dataset) with the function "get_df"
    df_final = pd.DataFrame()
    #df_base = get_df(data, cell_type, mapping, sample_type)
    
    ### applying the 5 clocks one-by-one
    for j in range(len(models_celltype)):
        ### taking one clock from the dataframe of all 5 clocks
        current_model = models_celltype.iloc[j].to_frame().T.reset_index()
        current_model = current_model.drop("index",axis=1)
        
        ### taking only the non-zero coefficients from the clock
        m2 = (abs(current_model) > 0).any()
        a = m2.index[m2]
        model_to_use = current_model[list(a)]
        
        ### applying "apply_clock" with the current clock
        pred, ages, donors, cell_names = apply_clock(df_base, impute_data, model_to_use, sample_type)
        
        ### adding the predictions given by the current clock to the final dataframe, with donor and age info
        df = pd.DataFrame()
        df["donor"] = donors
        df["age"] = ages
        df["predicted_age"] = pred
        df_final = pd.concat([df_final, df])
        
    ### writing the final datarframe into a csv file; this can be changed if needed
    df_final.to_csv("./application_of_clocks/"+dataset_name+".csv", index=False)
    
    
    
### running the prediction process

print("Start")
i=1
sample_type="pb"
pb_type="whole"
print("pb_whole")
imputation_file = "pseudobulk.csv"
df_base = get_df(adata, mapping_dict, sample_type, pb_type)
for param in [0.5]:
    clock_file = "pseudobulk_"+str(param)+".csv"
    predict(df_base, "pseudobulk_"+str(param), mapping_dict, clock_file, imputation_file, sample_type)
    print(str(i)+"/"+str(3))
    i+=1

i=1
sample_type="pb_sampled500"
pb_type="whole"
print("pb_sampled500")
imputation_file = "pseudobulk_bootstrapped_500.csv"
df_base = get_df(adata, mapping_dict, sample_type, pb_type)
for param in [0.5]:
    clock_file = "pseudobulk_bootstrapped_500_"+str(param)+".csv"
    predict(df_base, "pseudobulk_bootstrapped500_"+str(param), mapping_dict, clock_file, imputation_file, sample_type)
    print(str(i)+"/"+str(3))
    i+=1
    
i=1
sample_type="pb_sampled100"
pb_type="whole"
print("pb_sampled100")
imputation_file = "pseudobulk_bootstrapped_100.csv"
df_base = get_df(adata, mapping_dict, sample_type, pb_type)
for param in [0.5]:
    clock_file = "pseudobulk_bootstrapped_100_"+str(param)+".csv"
    predict(df_base, "pseudobulk_bootstrapped100_"+str(param), mapping_dict, clock_file, imputation_file, sample_type)
    print(str(i)+"/"+str(3))
    i+=1
    
i=1
sample_type="pb"
pb_type="glia"
print("glia_whole")
imputation_file = "pseudobulk_glia.csv"
df_base = get_df(adata, mapping_dict, sample_type, pb_type)
for param in [0.5]:
    clock_file = "pseudobulk_glia_"+str(param)+".csv"
    predict(df_base, "pseudobulk_glia_"+str(param), mapping_dict, clock_file, imputation_file, sample_type)
    print(str(i)+"/"+str(3))
    i+=1
    
i=1
sample_type="pb_sampled500"
pb_type="glia"
print("glia_sampled500")
imputation_file = "pseudobulk_glia_bootstrapped_500.csv"
df_base = get_df(adata, mapping_dict, sample_type, pb_type)
for param in [0.5]:
    clock_file = "pseudobulk_glia_bootstrapped_500_"+str(param)+".csv"
    predict(df_base, "pseudobulk_glia_bootstrapped500_"+str(param), mapping_dict, clock_file, imputation_file, sample_type)
    print(str(i)+"/"+str(3))
    i+=1
    
i=1
sample_type="pb_sampled100"
pb_type="glia"
print("glia_sampled100")
imputation_file = "pseudobulk_glia_bootstrapped_100.csv"
df_base = get_df(adata, mapping_dict, sample_type, pb_type)
for param in [0.5]:
    clock_file = "pseudobulk_glia_bootstrapped_100_"+str(param)+".csv"
    predict(df_base, "pseudobulk_glia_bootstrapped100_"+str(param), mapping_dict, clock_file, imputation_file, sample_type)
    print(str(i)+"/"+str(3))
    i+=1
    
i=1
sample_type="pb"
pb_type="neuron"
print("neuron_whole")
imputation_file = "pseudobulk_neuron.csv"
df_base = get_df(adata, mapping_dict, sample_type, pb_type)
for param in [0.5]:
    clock_file = "pseudobulk_neuron_"+str(param)+".csv"
    predict(df_base, "pseudobulk_neuron_"+str(param), mapping_dict, clock_file, imputation_file, sample_type)
    print(str(i)+"/"+str(3))
    i+=1
    
i=1
sample_type="pb_sampled500"
pb_type="neuron"
print("neuron_sampled500")
imputation_file = "pseudobulk_neuron_bootstrapped_500.csv"
df_base = get_df(adata, mapping_dict, sample_type, pb_type)
for param in [0.5]:
    clock_file = "pseudobulk_neuron_bootstrapped_500_"+str(param)+".csv"
    predict(df_base, "pseudobulk_neuron_bootstrapped500_"+str(param), mapping_dict, clock_file, imputation_file, sample_type)
    print(str(i)+"/"+str(3))
    i+=1
    
i=1
sample_type="pb_sampled100"
pb_type="neuron"
print("neuron_sampled100")
imputation_file = "pseudobulk_neuron_bootstrapped_100.csv"
df_base = get_df(adata, mapping_dict, sample_type, pb_type)
for param in [0.5]:
    clock_file = "pseudobulk_neuron_bootstrapped_100_"+str(param)+".csv"
    predict(df_base, "pseudobulk_neuron_bootstrapped100_"+str(param), mapping_dict, clock_file, imputation_file, sample_type)
    print(str(i)+"/"+str(3))
    i+=1