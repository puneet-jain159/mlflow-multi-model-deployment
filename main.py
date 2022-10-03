# Databricks notebook source
import pandas as pd
import json
import time
import mlflow
from pyspark.sql import DataFrame, functions as F, Window
from pyspark.sql.functions import explode, col, when, row_number, upper, sqrt, abs, greatest, least, udf
from pyspark.sql.types import StructType, StructField, StringType, BooleanType, DateType, IntegerType, DoubleType, FloatType
import numpy as np
from pandas import json_normalize

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Load all the relevant models from the mlflow either using runs or model registry

# COMMAND ----------

##############################################
# Call IR estimation API
##############################################

run_id1 = "06439607c9e1468cb61a46f7daf5e44f"
model_uri = "runs:/" + run_id1 + "/IRresponseModel"

IRresponseModel_model = mlflow.pyfunc.load_model(model_uri)
  
##############################################
# Call CR estimation API
##############################################
  
run_id1 = "028969078cf241bc812a7fce1bc5c987"
model_uri = "runs:/" + run_id1 + "/CRResponseModel"

CRResponseModel_model = mlflow.pyfunc.load_model(model_uri)


##############################################
# Call LOI estimation API
##############################################
  
run_id1 = "0ce6adc6a5cb4a259f99bf11b642238b"
model_uri = "runs:/" + run_id1 + "/LOIresponseModel"

LOIresponseModel_model = mlflow.pyfunc.load_model(model_uri)

##########################################################
# ATHENA get price
##########################################################

run_id1 = "79f27840874546c99fad2c68206bb7d7"
model_uri = "runs:/" + run_id1 + "/GetPriceInLocalModel"

GetPriceInLocalModel_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### load relevant datasets used by the model

# COMMAND ----------

audience_country_commission = spark.sql(f"SELECT * from kantar_mlops.audience_country_commission").toPandas()
latest_currency_df = spark.sql("SELECT * FROM kantar_mlops.currency_conversion_rates WHERE month in (SELECT max(month) FROM kantar_mlops.currency_conversion_rates)").toPandas()

# COMMAND ----------

class EnsembleModel(mlflow.pyfunc.PythonModel):
    """
      Class to use the GetPriceInLocalModel
    """

    def __init__(self,
                 IRresponseModel_model= IRresponseModel_model,
                 LOIresponseModel_model = LOIresponseModel_model,
                 CRResponseModel_model = CRResponseModel_model,
                 GetPriceInLocalModel_model = GetPriceInLocalModel_model,
                 audience_country_commission = audience_country_commission,
                 latest_currency_df = latest_currency_df):
        self.IRresponseModel_model = IRresponseModel_model,
        self.LOIresponseModel_model = LOIresponseModel_model,
        self.CRResponseModel_model = CRResponseModel_model,
        self.GetPriceInLocalModel_model = GetPriceInLocalModel_model
        self.audience_country_commission = audience_country_commission
        self.latest_currency_df = latest_currency_df
  
    def _preprocess_input(self,data_input):
      '''
      Preprocess the data input 
      '''
      countries_explode = data_input.explode("countries")
      audiences_explode = json_normalize(countries_explode['countries']).explode("audiences")['audiences'].apply(pd.Series)

      expanded_pd = pd.json_normalize(Sugar_json)\
      .join(countries_explode['countries'].apply(pd.Series))\
      .join(audiences_explode)\
      .drop(columns=['countries', 'audiences'], axis=1)

      expanded_pd.rename({'total_costs.incentives': 'incentives_costs', 
           'total_costs.panel_partner': 'panel_partner_costs',
           'total_costs.labor': 'labor_costs',
           'total_costs.translations': 'translations_costs',
           'total_costs.other': 'other_costs',
           'total_costs.panel_investment': 'panel_investment_costs',
           'account.account_uuid': 'account_uuid',
           'account.account_name': 'account_name',
           'account.client_strategic_index': 'client_strategic_index',
           'account.client_segment': 'client_segment',
           'account.country_code': 'account_country_code',
           'account.company_number': 'account_company_number',
           'account.maconomy_number': 'account_maconomy_number',
           'account.preferred_currency': 'account_preferred_currency',
           'country_code': 'proposal_country_code',
           'completes': 'audience_completes',
           'incidence_rate': 'audience_incidence_rate',
           'length_of_interview': 'audience_length_of_interview',
           'panel_source': 'audience_panel_source',
           'panelist_type': 'audience_panelist_type',
           'sampling_complexity': 'audience_sampling_complexity',
           'waves': 'audience_waves'}, axis=1, inplace=True)

      return expanded_pd
    
    
    def _call_athena_models(self,expanded_pd):
        '''
        Function to call the models and get the input
        '''
        #preprocess features before applying model
        features_df = expanded_pd[['proposal_country_code', 'account_company_number', 'client_segment', 'service_type', 'audience_panelist_type', 
                                   'audience_sampling_complexity','audience_length_of_interview', 'proposed_fieldwork_length', 'audience_waves',
                                   'audience_completes', 'audience_incidence_rate', 'account_country_code' ]]

        features_df.rename ({'proposal_country_code': 'AudienceCountryA3',
                            'account_company_number': 'AccountingCompanyNum',
                             'client_segment': 'GlobalClientSegment',
                             'service_type': 'OptionServiceType',
                             'audience_panelist_type': 'feaAudiencePanelistType', 
                             'audience_sampling_complexity': 'fea_SugarSamplingComplexity_v2',
                             'audience_length_of_interview': 'SoldLOI',
                             'proposed_fieldwork_length': 'feaFieldworkLength',
                             'audience_waves': 'feaAudienceWaves',
                             'audience_completes': 'PrmReqCompletes',
                             'audience_incidence_rate': 'PrmWeightedIR_1'}, axis=1, inplace=True)

        features_df['feaNumberOfAudienceCountryA3'] = features_df['AudienceCountryA3'].map(features_df['AudienceCountryA3'].value_counts())
        features_df['feaAudienceCountryIsAccountCountry'] = (features_df['AudienceCountryA3']==features_df['account_country_code']).astype(int)

        # Get response from the model

        IR_response = IRresponseModel_model.predict(features_df)
        CR_response =  CRResponseModel_model.predict(features_df)
        LOI_response = LOIresponseModel_model.predict(features_df)

        df1 = expanded_pd.join(pd.DataFrame(IR_response, columns=['Athena_IR']).astype(int))
        df2 = df1.join((pd.DataFrame(CR_response, columns=['Athena_CR'])*100).astype(int))
        df3 = df2.join(pd.DataFrame(LOI_response, columns=['Athena_LOI']).astype(int))

        return df3

    def _score_data_apply_post_processing(self,expanded_pd):
        '''
        Score models on the data
        '''
        scores = self._call_athena_models(expanded_pd)
        model_output_results = pd.DataFrame(scores).reset_index().rename(columns={"index":"row_id"})

        MinIRG = 2
        MinIRA = 2
        MaxIRA = 100
        MaxLOI = 120
        MinLOI = 1

        model_output_results.loc[model_output_results['Athena_IR']<=MinIRG , 'Athena_IR'] = MinIRG
        model_output_results.loc[model_output_results['Athena_IR']<=MinIRA , 'Athena_IR'] = MinIRA
        model_output_results.loc[model_output_results['Athena_IR']>=MaxIRA , 'Athena_IR'] = MaxIRA
        model_output_results.loc[model_output_results['Athena_LOI']<=MinLOI , 'Athena_LOI'] = MinLOI
        model_output_results.loc[model_output_results['Athena_LOI']>=MaxLOI , 'Athena_LOI'] = MaxLOI

        return model_output_results
  
    def _apply_dave_model(self,model_output_results):
      '''
      Applying Dave's model and post processing the resutls
      '''
      A_df = model_output_results[['proposal_country_code','account_country_code','project_type','brand','account_company_number'
                             ,'Athena_IR','Athena_LOI',"audience_incidence_rate","audience_length_of_interview"]]
      A_df.rename ({'account_company_number': 'company',
                          'project_type': 'product'}, axis=1, inplace=True)
      A_df['country'] = '(All)'
      A_df['segment'] = '(All)'
      A_df['client'] = '(All)'
      A_df['loi'] = A_df['Athena_IR']/100
      A_df['jobIr'] = A_df['Athena_LOI']
      A_df['srcSystem'] = 'Sugar'

      F_df = A_df.merge(self.audience_country_commission[['Audience_Country_A3','Company']], how = 'inner', 
                                             left_on='proposal_country_code', right_on='Audience_Country_A3')

      F_df.drop('company',axis =1 ,inplace = True)
      F_df.rename({'Company' : 'company'},axis =1 ,inplace = True)
      
      ##########################################################
      # ATHENA get price
      ##########################################################

      A_Results = self.GetPriceInLocalModel_model.predict(A_df)
      F_Results = self.GetPriceInLocalModel_model.predict(F_df)
      
      ##########################################################
      # Sold get price
      ##########################################################
      A_sold_df = A_df.copy()
      A_sold_df['loi'] = A_sold_df['audience_incidence_rate']/100
      A_sold_df['jobIr'] = A_sold_df['audience_length_of_interview']

      F_sold_df = F_df.copy()
      F_sold_df['loi'] = F_sold_df['audience_incidence_rate']/100
      F_sold_df['jobIr'] = F_sold_df['audience_length_of_interview']

      A_SoldResults = GetPriceInLocalModel_model.predict(A_sold_df)
      F_SoldResults = GetPriceInLocalModel_model.predict(F_sold_df)
      
         
      #################################################################
      # Fieldwork Country
      #################################################################

      F_Results_df = pd.DataFrame(F_Results, columns =["Fieldwork_Athena_CPI","Athena_IR",
                                                       "Athena_LOI","Fieldwork_Currency"]).reset_index().rename(columns={"index":"row_id"})
      F_Results_df['Fieldwork_Athena_CPI'] = F_Results_df['Fieldwork_Athena_CPI'].astype('float')
      F_Results_df['Fieldwork_Currency'] = F_Results_df['Fieldwork_Currency'].str.upper()
      F_Results_df = F_Results_df.merge(self.latest_currency_df, how = 'left', left_on='Fieldwork_Currency', right_on='From_Currency')
      F_Results_df['Rate'] = F_Results_df['Rate'].astype('float')
      F_Results_df['Fieldwork_Athena_CPI_USD'] = F_Results_df['Fieldwork_Athena_CPI']*F_Results_df['Rate']

      #################################################################
      # Account Country
      #################################################################

      A_Results_df = pd.DataFrame(A_Results, columns =["Account_Athena_CPI","Athena_IR",
                                                       "Athena_LOI","Account_Currency"]).reset_index().rename(columns={"index":"row_id"})
      A_Results_df['Account_Athena_CPI'] = A_Results_df['Account_Athena_CPI'].astype('float')
      A_Results_df['Account_Currency'] = A_Results_df['Account_Currency'].str.upper()
      A_Results_df = A_Results_df.merge(self.latest_currency_df, how = 'left', left_on='Account_Currency', right_on='From_Currency')
      A_Results_df['Rate'] = A_Results_df['Rate'].astype('float')
      A_Results_df['Account_Athena_CPI_USD'] = A_Results_df['Account_Athena_CPI']*A_Results_df['Rate']
      A_Results_df = A_Results_df[['row_id','Account_Currency','Account_Athena_CPI', 'Account_Athena_CPI_USD']]

      ATHENA_Price_df  = F_Results_df.merge(A_Results_df, how ='left', left_on='row_id', right_on='row_id')
      ATHENA_Price_df =  ATHENA_Price_df[['row_id','Athena_IR','Athena_LOI','Fieldwork_Currency','Fieldwork_Athena_CPI',
                                           'Fieldwork_Athena_CPI_USD','Account_Currency','Account_Athena_CPI','Account_Athena_CPI_USD']]
      
      
      #################################################################
      # Fieldwork Country
      #################################################################

      F_SoldResults_df = pd.DataFrame(F_SoldResults, columns =["Fieldwork_Sold_CPI","Sold_IR","Sold_LOI","Fieldwork_Currency"]).reset_index().rename(columns={"index":"row_id"})
      F_SoldResults_df['Fieldwork_Sold_CPI'] = F_SoldResults_df['Fieldwork_Sold_CPI'].astype('float')
      F_SoldResults_df['Fieldwork_Currency'] = F_SoldResults_df['Fieldwork_Currency'].str.upper()
      F_SoldResults_df = F_SoldResults_df.merge(self.latest_currency_df, how = 'left', left_on='Fieldwork_Currency', right_on='From_Currency')
      F_SoldResults_df['Rate'] = F_SoldResults_df['Rate'].astype('float')
      F_SoldResults_df['Fieldwork_Sold_CPI_USD'] = F_SoldResults_df['Fieldwork_Sold_CPI']*F_Results_df['Rate']

      #################################################################
      # Account Country
      #################################################################

      A_SoldResults_df = pd.DataFrame(A_SoldResults, columns =["Account_Sold_CPI","Sold_IR","Sold_LOI","Account_Currency"]).reset_index().rename(columns={"index":"row_id"})
      A_SoldResults_df['Account_Sold_CPI'] = A_SoldResults_df['Account_Sold_CPI'].astype('float')
      A_SoldResults_df['Account_Currency'] = A_SoldResults_df['Account_Currency'].str.upper()
      A_SoldResults_df = A_SoldResults_df.merge(self.latest_currency_df, how = 'left', left_on='Account_Currency', right_on='From_Currency')
      A_SoldResults_df['Rate'] = A_SoldResults_df['Rate'].astype('float')
      A_SoldResults_df['Account_Sold_CPI_USD'] = A_SoldResults_df['Account_Sold_CPI']*A_SoldResults_df['Rate']
      A_SoldResults_df = A_SoldResults_df[['row_id','Account_Sold_CPI','Account_Sold_CPI_USD']]

      Sold_Price_df  = F_SoldResults_df.merge(A_SoldResults_df, how ='left', left_on='row_id', right_on='row_id')
      Sold_Price_df = Sold_Price_df[['row_id', 'Sold_IR','Sold_LOI','Fieldwork_Sold_CPI_USD','Account_Sold_CPI','Account_Sold_CPI_USD','Rate']]
      # combine the df
      return  ATHENA_Price_df.merge(Sold_Price_df, how ='left', left_on='row_id', right_on='row_id')
    
  
    def _apply_adjudication(self,df,model_output_results):
      '''
      applying the balancing Adjudication
      '''
      df['P2'] = df[["Fieldwork_Athena_CPI_USD", "Account_Athena_CPI_USD"]].max(axis=1)
      df['P1'] = df[["Fieldwork_Sold_CPI_USD", "Account_Sold_CPI_USD"]].max(axis=1)
      df['LOI1'] = df[["Sold_LOI"]]
      df['LOI2'] = df[["Athena_LOI"]]
      df['IR1'] = df[["Sold_IR"]]*100
      df['IR1'] = df[['IR1']].round(0).astype(int)
      df['IR2'] = df[["Athena_IR"]]*100
      df['IR2'] = df[['IR2']].round(0).astype(int)

      dave_mor_df = df[['row_id','P2','P1','LOI1','LOI2','IR1','IR2']]

      # Read from station 5 table (entity)
      # add Version_TS
      # when query we pick the max timestamp

      # below are sample values for building logic
      BetaC = 0.9
      BetaA = 0.8
      BetaB = 0.5
      # Control parameters kept inside modeling pipeline
      a = 0.05
      b = 0.04
      c = 0.03
      d = 0.02
      alpha = 0.65

      DefaultIndex = 3
      model_output_results.loc[model_output_results['proposal_strategic_index']<=DefaultIndex , 'proposal_strategic_index'] = DefaultIndex
      model_output_results.loc[model_output_results['competitive_intensity_index']<=DefaultIndex , 'competitive_intensity_index'] = DefaultIndex
      model_output_results.loc[model_output_results['client_strategic_index']<=DefaultIndex , 'client_strategic_index'] = DefaultIndex

      adj_test = dave_mor_df.merge(model_output_results, how ='left', left_on='row_id', right_on='row_id')[["row_id", "P2", "P1", "LOI1", "LOI2",
                                                                                                            "IR1", "IR2","client_strategic_index", "proposal_strategic_index", "competitive_intensity_index"]].rename(columns={"client_strategic_index": "CSI", "proposal_strategic_index": "PSI","competitive_intensity_index":"CII"})

      adj_test['S'] = np.sqrt((adj_test['P2'] - adj_test['P1']).abs())/ (alpha *(adj_test['LOI1'] - adj_test['LOI2']).abs() + (adj_test['IR1'] - adj_test['IR2']).abs())
      adj_test['indox_combo'] = (a*adj_test["S"] + b*(adj_test["CSI"] - 3) + c*(adj_test["CII"] - 3) + d*(adj_test["PSI"] - 3))
      adj_test['gamma_temp_0'] = adj_test['indox_combo'].where(adj_test['indox_combo']<0, 0)

      adj_test["gammaA"] = 1 / (1+(1-BetaA)*adj_test['gamma_temp_0']) 
      adj_test["gammaB"] = 1 / (1+(1-BetaB)*adj_test['gamma_temp_0']) 
      adj_test["gammaC"] = 1 / (1+(1-BetaC)*adj_test['gamma_temp_0']) 

      adj_test["PriceA"] = adj_test["gammaA"] * adj_test["P2"] + (1-adj_test["gammaA"])*adj_test["P1"]
      adj_test["PriceB"] = adj_test["gammaB"] * adj_test["P2"]  + (1-adj_test["gammaB"])*adj_test["P1"]
      adj_test["PriceC"] = adj_test["gammaC"] * adj_test["P2"]  + (1-adj_test["gammaC"])*adj_test["P1"]

      adj_test["HighPriceUSD"] = adj_test[["PriceA", "PriceB","PriceC"]].max(axis=1)
      adj_test["LowPriceUSD"] = adj_test[["PriceA", "PriceB","PriceC"]].min(axis=1)
      adj_test["Strand2PriceUSD"] = adj_test[["PriceA", "PriceB","PriceC"]].median(axis=1)

      return model_output_results.merge(adj_test, how ='left', left_on='row_id', right_on='row_id')[['proposal_id','proposal_uuid',
                                                                              'revision_number','proposal_country_code','audience_uuid',
                                                                              'currency_code','HighPriceUSD','LowPriceUSD','Strand2PriceUSD']]
      
      
    def _apply_currency_conversion(self,output_df):
      '''
      Apply currency conversion to get the result
      '''
      output_json_df = output_df.merge(self.latest_currency_df, how = 'left', left_on='currency_code', right_on='From_Currency')
      output_json_df['Rate'] = output_json_df['Rate'].astype('float')
      output_json_df["HighPrice"] = output_json_df["HighPriceUSD"]* output_json_df["Rate"]
      output_json_df["LowPrice"] = output_json_df["LowPriceUSD"]* output_json_df["Rate"]
      output_json_df["Strand2Price"] = output_json_df["Strand2PriceUSD"]* output_json_df["Rate"]
      output_json_df = output_json_df[['proposal_id','proposal_uuid','revision_number','proposal_country_code',
                                       'audience_uuid','currency_code','HighPrice','LowPrice','Strand2Price']]
      
      return output_json_df
    
    
    def predict(self,context,data_input):
      '''
      Final method called by the API
      '''
      expanded_pd = Ensemble._preprocess_input(data_input)
      model_output_results = Ensemble._score_data_apply_post_processing(expanded_pd)
      df = Ensemble._apply_dave_model(model_output_results)
      output_df = Ensemble._apply_adjudication(df,model_output_results)
      output_df = Ensemble._apply_currency_conversion(output_df)
      return output_df.to_json(orient="records")

# COMMAND ----------

Ensemble = EnsembleModel(IRresponseModel_model= IRresponseModel_model,
                         LOIresponseModel_model = LOIresponseModel_model,
                         CRResponseModel_model = CRResponseModel_model,
                         GetPriceInLocalModel_model = GetPriceInLocalModel_model,
                         audience_country_commission = audience_country_commission,
                         latest_currency_df = latest_currency_df)

with mlflow.start_run():
  mlflow.pyfunc.log_model("EnsembleModel",
                   python_model = Ensemble,
                   pip_requirements = ['numpy','pandas','mlflow'])

# COMMAND ----------

Sugar_json = [{
    "proposal_id":1234,
    "proposal_uuid":"a4c8e12a-f778-11ec-8e94-0242ac110002",
    "proposal_name":"Test proposal",
    "revision_number":3,
    "pricing_formula":"2022-04",
    "brand":"External", #new
    "proposed_fieldwork_length":20,
    "fieldwork_start_date":"08/23/2022",
    "fieldwork_end_date":"09/12/2022",
    "service_type":"Sample only",
    "project_type":"Tracker",
    "rush_job": True, 
    "proposal_strategic_index":3,
    "competitive_intensity_index":4,
    "currency_code":"USD",
    "total_price":2832.03,
    "total_costs":{
        "incentives":145.02,
        "panel_partner":186.02,
        "labor":384.43,
        "translations":0.00,
        "other":0.00,
        "panel_investment":1073.33
    },
    "account":{
        "account_uuid":"a4c8e12a-f778-11ec-8e94-0242ac110002",
        "account_name":"Test account",
        "client_strategic_index":1,
        "client_segment":"Enterprise",
        "country_code":"USA",
        "company_number":"91",
        "maconomy_number":"91292102",
        "preferred_currency":"USD"    
    },
    "countries":[
        {
            "country_code":"FRA",
            "audiences":[
                {
                    "audience_uuid":"a4c8e12a-f778-11ec-8e94-0242ac110002",
                    "audience_name":"Main",
                    "completes":1000,
                    "incidence_rate":80,
                    "length_of_interview":12,
                    "panel_source":"LifePoints",
                    "panelist_type":"Consumer",
                    "sampling_complexity":"Low",
                    "audience_complexity_index":3,
                    "waves":3
                },
                {
                    "audience_uuid":"asdfa4c8e12a-f778-11ec-8e94-0242ac110002asdF",
                    "audience_name":"Boost",
                    "completes":200,
                    "incidence_rate":30,
                    "length_of_interview":12,
                    "panel_source":"LifePoints",
                    "panelist_type":"Consumer",
                    "sampling_complexity":"Low",
                    "audience_complexity_index":5,
                    "waves":3
                }
            ]
        },
        {
            "country_code":"DEU", #changed from GER
            "audiences":[
                {
                    "audience_uuid":"a4c8e12a-f778-11ec-8e94-0242ac110002",
                    "audience_name":"Main",
                    "completes":1000,
                    "incidence_rate":80,
                    "length_of_interview":12,
                    "panel_source":"LifePoints",
                    "panelist_type":"Consumer",
                    "sampling_complexity":"Low",
                    "audience_complexity_index":3,
                    "waves":3
                },
                {
                    "audience_uuid":"a4c8e12a-f778-11ec-8e94-0242ac110002",
                    "audience_name":"Boost",
                    "completes":200,
                    "incidence_rate":30,
                    "length_of_interview":12,
                    "panel_source":"LifePoints",
                    "panelist_type":"Consumer",
                    "sampling_complexity":"Low",
                    "audience_complexity_index":5,
                    "waves":3
                }
            ]
        }
    ]
}]

# COMMAND ----------

Ensemble = EnsembleModel(IRresponseModel_model= IRresponseModel_model,
                         LOIresponseModel_model = LOIresponseModel_model,
                         CRResponseModel_model = CRResponseModel_model,
                         GetPriceInLocalModel_model = GetPriceInLocalModel_model,
                         audience_country_commission = audience_country_commission,
                         latest_currency_df = latest_currency_df)

expanded_pd = Ensemble._preprocess_input(pd.DataFrame(Sugar_json))
model_output_results = Ensemble._score_data_apply_post_processing(expanded_pd)
df = Ensemble._apply_dave_model(model_output_results)
output_df = Ensemble._apply_adjudication(df,model_output_results)
output_df = Ensemble._apply_currency_conversion(output_df)
output_df.to_json(orient="records")

# COMMAND ----------

json.dumps(pd.DataFrame(Sugar_json).to_dict(orient='records'), allow_nan=True)

# COMMAND ----------

##############################################
# Call IR estimation API
##############################################

run_id1 = "b66dcb0a55d6403fba004b87d8f59a89"
model_uri = "runs:/" + run_id1 + "/EnsembleModel"

EnsembleModel = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

EnsembleModel.predict(pd.DataFrame(Sugar_json))

# COMMAND ----------


