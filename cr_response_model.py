# Databricks notebook source
import mlflow
import numpy as np 

# COMMAND ----------

class CRResponseModel(mlflow.pyfunc.PythonModel):
    """
      Class to use HuggingFace Models
    """

    def __init__(self, model=None):
        self.model = model

    def predict(self,context, model_input):
        '''
        Mock the return function here
        '''
        rng = np.random.default_rng(1653)
        rints = rng.integers(low=0, high=1, size=len(model_input))
        return rints


# COMMAND ----------

# MAGIC %md
# MAGIC #### Create the custom model here the can be called from the other places

# COMMAND ----------

with mlflow.start_run():
  mlflow.pyfunc.log_model("CRResponseModel",
                   python_model = CRResponseModel(),
                   pip_requirements = ['numpy','mlflow'])

# COMMAND ----------



# COMMAND ----------


