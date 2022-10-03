# Databricks notebook source
import requests
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

def getAkey():
  '''
  Function to load a snapshot of the table
  '''
  price_factors = spark.table('kantar_mlops.price_factors')

  theTable = price_factors \
  .withColumn('key=rgnId,brandId,ProductId,countryId,LOI,ClientSegment,Client', F.concat(F.col('RegionID'),F.col('BrandID'),F.col('ProductID'),F.col('CountryID'),F.col('LOI'),F.col('Client_Segment'),F.col('Client')))

  aKey = theTable.toPandas().set_index("key=rgnId,brandId,ProductId,countryId,LOI,ClientSegment,Client")
  aKey['Low_IR_Used'] = aKey['Low_IR_Used'].astype('float')
  aKey['Mid_IR_Used'] = aKey['Mid_IR_Used'].astype('float')
  aKey['High_IR_Used'] = aKey['High_IR_Used'].astype('float')
  aKey['Low_IR_Anchor'] = aKey['Low_IR_Anchor'].astype('float')
  aKey['Mid_IR_Anchor'] = aKey['Mid_IR_Anchor'].astype('float')
  aKey['High_IR_Anchor'] = aKey['High_IR_Anchor'].astype('float')
  aKey['xFactor_Low_IR'] = aKey['xFactor_Low_IR'].astype('float')
  aKey['xFactor_Mid_IR'] = aKey['xFactor_Mid_IR'].astype('float')
  aKey['xFactor_High_IR'] = aKey['xFactor_High_IR'].astype('float')
  aKey['Low_IR_Step'] = aKey['Low_IR_Step'].astype('float')
  aKey['Mid_IR_Step'] = aKey['Mid_IR_Step'].astype('float')
  aKey['High_IR_Step'] = aKey['High_IR_Step'].astype('float')
  aKey['Break1'] = aKey['Break1'].astype('float')
  aKey['Break2'] = aKey['Break2'].astype('float')
  aKey['Break3'] = aKey['Break3'].astype('float')
  aKey['Break4'] = aKey['Break4'].astype('float')
  aKey['Break5'] = aKey['Break5'].astype('float')
  aKey['Break6'] = aKey['Break6'].astype('float')
  return aKey

# COMMAND ----------

class GetPriceInLocal:

  def __init__(self, company,brand,product,country,loi, segment, client, jobIr, srcSystem,aKey):
    self._company=company
    self.brand=brand
    self.product=product
    self.country=country
    self.loi=str(min(loi if loi % 5==0 else int(loi/5+1)*5,60))
    self.segment=segment
    self.client=client
    self.jobIr=jobIr
    self.srcSystem=srcSystem
    self.aKey = aKey
   #testJobIr=self.jobIr  # this id the IR passed in
  
  def __str__(self):
    # print default string of inbound paramaters
    return "thePrice =%d ,company = %s, brand=%s, product=%s, country=%s, loi=%s, segment=%s, client=%s, srcSystem=%s" %(self.thePrice,self._company, self.brand, 
                                                                                             self.product, self.country, self.loi, self.segment, self.client, self.srcSystem)
  
  # setup properties for each of the inbound items to test validity and/or set to default value
  @property
  def company(self):
    return self._company
  
  @company.setter
  # need to have a table pass in a list of valid companies going to cheat by creating an array
  def company(self,value):
    validCompanies=[13,36,37,80,91,92,93,105,112,113,121,145,201,233,243,250,283 ]
    for i in validCompaines:
      if value==i:
        self._company=value
      else:
        print ('invalid company number')     
      
  def _getFactors(self, theKey):
    return self.aKey.loc[theKey] 

  def _detAnchor_IR_Step_to_use(self, jobIR,Low_IR_Used, midIrUsed,highIrUsed,lowIrAnchor, midIrAnchor, highIrAnchor, xFactorLowIr, xFactorMidIr, xFactorHighIr,lowIrStep, midIrStep,
         highIrStep):
    # determine which IR point is used for the anchor and xFactor
    if jobIR>midIrUsed:
      return highIrUsed, highIrAnchor,xFactorHighIr, highIrStep
    elif jobIR>Low_IR_Used:
      return midIrUsed, midIrAnchor, xFactorMidIr,midIrStep
    else:
      return Low_IR_Used,lowIrAnchor,xFactorLowIr,lowIrStep
    
    return 'error on ir test'

  def _detBreakpt(self, jobIR,breakPoints):
    # need to read breakpoints and compare to passed-in Ir
    #......this needs to be built out to the 6
    if breakPoints[0]>1:
      return jobIR
    for i in range(0,5):
      if breakPoints[i]<jobIR:
        return breakPoints[i]
    return jobIR

  def _getIRfactor(self, irEvaluated,irStep):
    irFactor=1/int(irEvaluated/irStep+1)
    return irFactor

  def _getPrice(self, jobIRfactor,anchorIRfactor, anchorPrice,xFactor):
    priceLocal=(jobIRfactor/anchorIRfactor)**xFactor*anchorPrice
    return priceLocal

  def main(self):
     #create key to get factors based on constructor inputs
    theKey="".join([self.company,self.brand,self.product,self.country,self.loi,self.segment,self.client]) # '92ExternalAd Hoc(All)15(All)(All)'
    
    theKeyFactors = self._getFactors(theKey)
    # pass the values to determine which ir anchor is to be used and the relavant factors associated
    factorsToUse = \
    self._detAnchor_IR_Step_to_use(jobIR=self.jobIr,Low_IR_Used=theKeyFactors['Low_IR_Used'],midIrUsed=theKeyFactors['Mid_IR_Used'],highIrUsed=theKeyFactors['High_IR_Used'],\
                          lowIrAnchor=theKeyFactors['Low_IR_Anchor'],midIrAnchor=theKeyFactors['Mid_IR_Anchor'],highIrAnchor=theKeyFactors['High_IR_Anchor'],\
                          xFactorLowIr=theKeyFactors['xFactor_Low_IR'],xFactorMidIr=theKeyFactors['xFactor_Mid_IR'],xFactorHighIr=theKeyFactors['xFactor_High_IR'],\
                          lowIrStep=theKeyFactors['Low_IR_Step'],midIrStep=theKeyFactors['Mid_IR_Step'],highIrStep=theKeyFactors['High_IR_Step'])
    #get the breakpoints to see if we use the job ir or the breakpoint price
    breakPoints=theKeyFactors['Break1'],theKeyFactors['Break2'],theKeyFactors['Break3'],theKeyFactors['Break4'],theKeyFactors['Break5'],theKeyFactors['Break6']
    #print(breakPoints)
    # take the job ir passed in and compare to the breakpoints to determine which to use for pricing
    irToUse=self._detBreakpt(jobIR=self.jobIr,breakPoints=breakPoints)
    #print(irToUse)
    jobIrFactor =  self._getIRfactor(irToUse,factorsToUse[3])
    anchorIrFactor = self._getIRfactor(factorsToUse[0],factorsToUse[3])
    thePrice=self._getPrice(jobIRfactor=jobIrFactor,anchorIRfactor=anchorIrFactor,anchorPrice=factorsToUse[1],xFactor=factorsToUse[2])
    return [float(round(thePrice,2)),irToUse, int(self.loi), theKeyFactors['Rate_Currency']]

# COMMAND ----------

class GetPriceInLocalModel(mlflow.pyfunc.PythonModel):
    """
      Class to use the GetPriceInLocalModel
    """

    def __init__(self, aKey=aKey,GetPriceInLocal= GetPriceInLocal):
        self.aKey = aKey

    def predict(self,context, model_input):
        '''
        Add the function here
        '''
        pred = []
        for index, row in model_input.iterrows():
          Pricel = GetPriceInLocal(company = row['company'],
                                   brand = row['brand'],
                                   product = row['product'],
                                   country= row['country'],
                                   loi = row['loi'],
                                   segment = row['segment'],
                                   client = row['client'],
                                   jobIr = row['jobIr'],
                                   srcSystem = row['srcSystem'],
                                   aKey = self.aKey)
          pred.append(Pricel.main())
        return pred

# COMMAND ----------

with mlflow.start_run():
  mlflow.pyfunc.log_model("GetPriceInLocalModel",
                   python_model = GetPriceInLocalModel(aKey = getAkey(),GetPriceInLocal=GetPriceInLocal),
                   pip_requirements = ['numpy','pandas','mlflow'])

# COMMAND ----------

getAkey().dtypes

# COMMAND ----------


