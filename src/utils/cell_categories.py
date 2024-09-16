import pandas as pd

all_unique_classes = ['Neuronal: GABAergic','Non-neuronal and Non-neural','Neuronal: Glutamatergic']
all_unique_subclasses = ['Chandelier','Sncg','Oligodendrocyte','L6 IT Car3','Pax6','L5/6 NP','Pvalb','Microglia-PVM','Sst Chodl','Astrocyte','L4 IT','L5 IT','L6 CT','L2/3 IT','OPC','L6 IT','Sst','L5 ET','L6b','Lamp5 Lhx6','Vip','Lamp5','VLMC','Endothelial']
all_unique_supertypes = ['Sst_13','Astro_5','SMC-SEAAD','L4 IT_1','L6 IT Car3_3','Sst_7','Astro_2','L5 IT_2','L5 ET_1','L2/3 IT_3','Vip_21','L6 IT_1','L2/3 IT_10','Vip_18','Sst_11','L6b_6','Micro-PVM_2_3-SEAAD','Lamp5_2','Chandelier_2','Oligo_2_1-SEAAD','Endo_1','L2/3 IT_12','Pvalb_7','OPC_2_2-SEAAD','OPC_2_1-SEAAD','Sst_10','L6b_1','L6 CT_4','Sst_20','Vip_6','Lamp5_Lhx6_1','L5 IT_6','Endo_3','L6 IT Car3_1','Sst_25','Sst_23','Pericyte_1','L6 CT_2','L2/3 IT_2','L5 ET_2','Vip_2','Oligo_1','Sncg_6','Lamp5_6','Micro-PVM_1','Pax6_1','Vip_4','Monocyte','Micro-PVM_2','L6b_2','Lamp5_5','L5 IT_5','Vip_23','L5/6 NP_4','Sst_5','OPC_1','Pvalb_14','L6 CT_1','Vip_16','Pvalb_15','Pvalb_2','Sncg_1','Pvalb_13','L6b_3','L6b_4','Lamp5_4','Oligo_3','Sncg_8','L5/6 NP_2','Pvalb_9','Pax6_3','L6 CT_3','Pvalb_12','L4 IT_2','Vip_19','Vip_12','L5/6 NP_1','Sst Chodl_2','L2/3 IT_7','L6 IT_2','Sncg_5','Pax6_2','VLMC_1','Astro_4','Sst Chodl_1','L5 IT_3','Pericyte_2-SEAAD','L5 IT_1','Lamp5_3','Chandelier_1','L6 IT Car3_2','OPC_2','Vip_14','L6b_5','Micro-PVM_3-SEAAD','Sst_3','Sncg_2','L2/3 IT_5','L2/3 IT_13','Sst_9','Pax6_4','Pvalb_6','Micro-PVM_2_1-SEAAD','L4 IT_3','L4 IT_4','L5/6 NP_6','Vip_15','Sst_4','Pvalb_1','Vip_5','Sst_2','Sncg_4','Astro_1','L2/3 IT_8','Vip_9','L5/6 NP_3','L2/3 IT_1','Sst_19','Sst_22','Pvalb_10','Sst_1','Vip_1','Astro_3','L2/3 IT_6','Astro_6-SEAAD','Pvalb_3','Vip_13','Pvalb_8','Oligo_2','Sncg_3','Endo_2','Pvalb_5','Oligo_4','Lamp5_1','L5 IT_7','Vip_11','Sst_12']

def retrieve_all_cell_types_categories(adata):
    #First we create a categorical dtype for each category
    classes_dtype = pd.CategoricalDtype(categories=all_unique_classes)
    subclasses_dtype = pd.CategoricalDtype(categories=all_unique_subclasses)
    supertype_dtype = pd.CategoricalDtype(categories=all_unique_supertypes)

    #Then we define our series of being that type. Here is where if a sample only has 162 subclasses, it will get the extra 2 for example. (total 164)
    classes_categorical = pd.Series(adata.obs['Class'], dtype=classes_dtype)
    subclasses_categorical = pd.Series(adata.obs['Subclass'], dtype=subclasses_dtype)
    supertype_categorical = pd.Series(adata.obs['Supertype'], dtype=supertype_dtype)

    #Create the Dataframe that will be used by pd.get_dummies()
    complete_cell_types_dict = {'Class':classes_categorical , 'Subclass': subclasses_categorical, 'Supertype':supertype_categorical}
    complete_cell_types_df = pd.DataFrame(complete_cell_types_dict)

    return complete_cell_types_df
