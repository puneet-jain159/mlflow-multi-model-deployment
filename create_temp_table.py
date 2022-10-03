# Databricks notebook source
# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS `kantar_mlops`

# COMMAND ----------

df = spark.read.csv('dbfs:/Users/puneet.jain@databricks.com/kantar/stg_strand1.price_factors.csv', header = True)

df.write \
  .format('delta') \
  .saveAsTable("kantar_mlops.price_factors")

# COMMAND ----------

df = spark.read.csv('dbfs:/Users/puneet.jain@databricks.com/kantar/cur_finance.currency_conversion_rates.csv', header = True)

df.write \
  .format('delta') \
  .saveAsTable("kantar_mlops.currency_conversion_rates")

# COMMAND ----------

df = spark.read.csv('dbfs:/Users/puneet.jain@databricks.com/kantar/stg_strand1.audience_country_commission.csv', header = True)

df.write \
  .format('delta') \
  .saveAsTable("kantar_mlops.audience_country_commission")

# COMMAND ----------

df

# COMMAND ----------


