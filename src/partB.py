import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings("ignore")

part_b = pd.read_csv("../provider_data/partb/Part-B.csv",encoding='ISO-8859-1',low_memory=False)
leie_ex = pd.read_csv("../provider_data/LEIE_Exclusion.csv",encoding='ISO-8859-1',low_memory=False)

part_b["Rndrng_NPI"] = part_b["ï»¿Rndrng_NPI"]
part_b=part_b.drop("ï»¿Rndrng_NPI",axis=1)

part_b = part_b.dropna(axis=0,thresh=2)
part_b = part_b.dropna(subset=["Rndrng_NPI"],axis=0)
part_b = part_b.dropna(subset=["Rndrng_Prvdr_Last_Org_Name"],axis=0)
part_b = part_b.dropna(subset=["Rndrng_Prvdr_First_Name"],axis=0)

leie_ex = leie_ex.dropna(subset=["NPI"],axis=0)
ex_npi = leie_ex[leie_ex["NPI"]!=0]["NPI"]
ex_lnames = leie_ex.dropna(subset=["LASTNAME"],axis=0)["LASTNAME"]
ex_fnames = leie_ex.dropna(subset=["FIRSTNAME"],axis=0)["FIRSTNAME"]
leie_ex["MIDNAME"] = leie_ex["MIDNAME"].apply(lambda row : row if row!= " " else 0)
ex_mi = leie_ex[leie_ex["MIDNAME"]!=0]["MIDNAME"]


print('PARTB Shape : ',part_b.shape)

ex_npi_unique = leie_ex.drop_duplicates("NPI")  # Remove duplicate NPI values
ex_npi_dict = dict(zip(ex_npi_unique['NPI'], ex_npi_unique['EXCLTYPE']))

part_b['Fraud'] = np.where(part_b['Rndrng_NPI'].isin(ex_npi_dict), 1, 0)
part_b['FraudType'] = np.where(part_b['Fraud'] == 1,part_b['Rndrng_NPI'].map(ex_npi_dict),0)

print(part_b["Fraud"].value_counts())

def convert_to_int(value):
    try:
        return int(value)
    except ValueError:
        return np.nan  

part_b['Rndrng_Prvdr_Zip5'] = part_b['Rndrng_Prvdr_Zip5'].apply(convert_to_int)
part_b = part_b[part_b["Rndrng_Prvdr_Cntry"] == 'US']
part_b["Rndrng_Prvdr_Mdcr_Prtcptg_Ind"] = part_b["Rndrng_Prvdr_Mdcr_Prtcptg_Ind"].apply(lambda row : 1 if row == 'Y' else 0)
part_b["Tot_Benes"] = part_b["Tot_Benes"].str.replace(',', '').astype(float)
part_b["Tot_Srvcs"] = part_b["Tot_Srvcs"].str.replace(',','').astype(float)
part_b["Tot_Sbmtd_Chrg"] = part_b["Tot_Sbmtd_Chrg"].replace('[\$,]', '', regex=True).astype(float)
part_b["Tot_Mdcr_Alowd_Amt"] = part_b["Tot_Mdcr_Alowd_Amt"].replace('[\$,]', '', regex=True).astype(float)
part_b["Tot_Mdcr_Pymt_Amt"] = part_b["Tot_Mdcr_Pymt_Amt"].replace('[\$,]', '', regex=True).astype(float)
part_b["Tot_Mdcr_Stdzd_Amt"] = part_b["Tot_Mdcr_Stdzd_Amt"].replace('[\$,]', '', regex=True).astype(float)
part_b["Bene_CC_CHF_Pct"] = part_b["Bene_CC_CHF_Pct"].str.rstrip('%').astype('float') / 100.0

feature_columns = ['Rndrng_Prvdr_Crdntls','Drug_Sprsn_Ind',
       'Drug_Tot_HCPCS_Cds', 'Drug_Tot_Benes', 'Drug_Tot_Srvcs',
       'Drug_Sbmtd_Chrg', 'Drug_Mdcr_Alowd_Amt', 'Drug_Mdcr_Pymt_Amt',
       'Drug_Mdcr_Stdzd_Amt', 'Med_Sprsn_Ind', 'Med_Tot_HCPCS_Cds',
       'Med_Tot_Benes', 'Med_Tot_Srvcs', 'Med_Sbmtd_Chrg',
       'Med_Mdcr_Alowd_Amt', 'Med_Mdcr_Pymt_Amt', 'Med_Mdcr_Stdzd_Amt',
                   'Bene_Age_LT_65_Cnt', 'Bene_Age_65_74_Cnt',
       'Bene_Age_75_84_Cnt', 'Bene_Age_GT_84_Cnt', 'Bene_Feml_Cnt',
       'Bene_Male_Cnt', 'Bene_Race_Wht_Cnt', 'Bene_Race_Black_Cnt',
       'Bene_Race_API_Cnt', 'Bene_Race_Hspnc_Cnt', 'Bene_Race_NatInd_Cnt',
       'Bene_Race_Othr_Cnt', 'Bene_Dual_Cnt', 'Bene_Ndual_Cnt',
       'Bene_CC_AF_Pct', 'Bene_CC_Alzhmr_Pct', 'Bene_CC_Asthma_Pct',
       'Bene_CC_Cncr_Pct', 'Bene_CC_CHF_Pct', 'Bene_CC_CKD_Pct',
       'Bene_CC_COPD_Pct', 'Bene_CC_Dprssn_Pct', 'Bene_CC_Dbts_Pct',
       'Bene_CC_Hyplpdma_Pct', 'Bene_CC_Hyprtnsn_Pct', 'Bene_CC_IHD_Pct',
       'Bene_CC_Opo_Pct', 'Bene_CC_RAOA_Pct', 'Bene_CC_Sz_Pct',
       'Bene_CC_Strok_Pct']



part_b[feature_columns] = part_b[feature_columns].apply(pd.to_numeric, errors='coerce')
part_b_imputed = part_b.copy()
imputer = KNNImputer()
print('Imputating the Dataset .... ')
part_b_imputed[feature_columns] = imputer.fit_transform(part_b_imputed[feature_columns])
part_b_imputed.to_csv("imputedKNN5.csv")
