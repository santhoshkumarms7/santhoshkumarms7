import pandas as pd
import numpy as np
import re
import string
from collections import Counter, OrderedDict
import statistics
from scipy import stats
from datetime import datetime
from sherlock.features.stats_helper import compute_stats
from sherlock.global_state import is_first
from functional import pseq, seq
import multiprocessing
from sherlock.features.bag_of_words import count_pattern_in_cells 

core_count = multiprocessing.cpu_count()  #1
size = 100

### Check if the  data is of Integer type or not
def checkInt(strs):
        
    ### If integer type then return else return 0
    if isinstance(strs,int):
        return 1
    elif isinstance(strs,float):
        return 0
    else:
        try:
            int(strs)
            return 1
        except:
            return 0

### Check if the  data is of Float type or not
def checkFloat(strs):
        
    ### If Float type then return else return 0
    if isinstance(strs,float):
        return 1
    else:
        try:
            if checkInt(strs):
                return 0
            strs = float(strs)
            if strs != np.inf:
                return 1
            else:
                return 0
        except:
            return 0

### Identify the ratio of alpha to numeric ratio
def alphaAndNumericMatch(value):
    
    value = str(value)
    charCount = len(re.findall(string = value,pattern='[a-zA-Z]'))
    numCount = len(re.findall(string = value,pattern='\d'))
    specialCharCount = len(re.findall(string=value,pattern='[!#&\'()*+\-/:;<=>?@[\\]^_`{|}~]'))
    
    if (charCount >0 or specialCharCount) and numCount>0:
        return 'alphanumeric'
    elif numCount > 0:
        return 'numeric'
    elif charCount > 0:
        return 'alpha'
    else:
        return 'others'

def intTypeData(col_values):

  int_type_data = [int(element) for element in col_values if checkInt(element)]
  int_type_ratio = len(int_type_data)/len(col_values)
  _median = np.median(int_type_data)

  try:
    mean_before_int = np.mean([len(str(val)) for val in int_type_data])
  except:
    mean_before_int = 0

  if len(int_type_data)>0:
    _mean, _variance, _skew, _kurtosis, _min, _max, _sum = compute_stats(int_type_data)
    return int_type_data,int_type_ratio,_mean, _median, _variance, _skew, _kurtosis, _min, _max, _sum, mean_before_int
  else:
    return int_type_data,int_type_ratio,0,0,0,0,-3,0,0,0,mean_before_int

def floatTypeData(col_values):

  float_type_data = [float(element) for element in col_values if checkFloat(element)]
  float_type_ratio = len(float_type_data)/len(col_values)
  _median = np.median(float_type_data)

  try:
    mean_before_float = np.mean([len(str(float(val)).split('.')[0]) for val in float_type_data if pd.notnull(val)])
    mean_after_float = np.mean([len(str(float(val)).split('.')[1]) for val in float_type_data if len(str(float(val)).split('.'))>1 and pd.notnull(val)])
    max_after_float = max([int(str(float(val)).split('.')[1]) for val in float_type_data if len(str(float(val)).split('.'))>1 and pd.notnull(val)])
  except:
    mean_before_float = 0
    mean_after_float = 0
    max_after_float = 0
    
  zero_flag = 1

  if max_after_float >0:
    zero_flag = 0

  if len(float_type_data)>0:
    _mean, _variance, _skew, _kurtosis, _min, _max, _sum = compute_stats(float_type_data)
    return float_type_data, float_type_ratio, _mean, _median, _variance, _skew, _kurtosis, _min, _max, _sum, mean_before_float, mean_after_float,max_after_float,zero_flag
  else:
    return float_type_data,float_type_ratio,0,0,0,0,-3,0,0,0,mean_before_float,mean_after_float,max_after_float,zero_flag

### Validate if the input data is of date/datetime format
def checkDate(strs):

  date_pattern = " \d{4}-[0-1][0-9]-[0-3][0-9] | \d{4}-[0-3][0-9]-[0-1][0-9] | [0-1][0-9]-\d{4}-[0-3][0-9] | [0-1][0-9]-[0-3][0-9]-\d{4} | [0-3][0-9]-[0-1][0-9]-\d{4} | [0-3][0-9]-\d{4}-[0-1][0-9] | \d{4}-(\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+-[0-3][0-9] | \d{4}-[0-3][0-9]-(\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+ | (\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+-\d{4}-[0-3][0-9] | (\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+-[0-3][0-9]-\d{4} | [0-3][0-9]-(\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+-\d{4} | [0-3][0-9]-\d{4}-(\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+ | \d{4}-(\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+-[0-3][0-9] | \d{4}-[0-3][0-9]-(\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+ | (\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+-\d{4}-[0-3][0-9] | (\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+-[0-3][0-9]-\d{4} | [0-3][0-9]-(\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+-\d{4} | [0-3][0-9]-\d{4}-(\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+ | \d{4}-(\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+ | \d{4}-(\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+ | (\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+-\d{4} | (\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+-\d{4} | [0-3][0-9]-(\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+ | [0-3][0-9]-(\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+ | (\\bjanuary\\b|\\bfebruary\\b|\\bmarch\\b|\\bapril\\b|\\bmay\\b|\\bjune\\b|\\bjuly\\b|\\baugust\\b|\\bseptember\\b|\\boctober\\b|\\bnovember\\b|\\bdecember\\b)+-[0-3][0-9] | (\\bjan\\b|\\bfeb\\b|\\bmar\\b|\\bapr\\b|\\bmay\\b|\\bjun\\b|\\bjul\\b|\\baug\\b|\\bsept\\b|\\boct\\b|\\bnov\\b|\\bdec\\b)+-[0-3][0-9] "
   
  strs = str(strs).replace('/',' ').replace(',',' ').replace('.',' ').replace(' ','-')
  strs = re.sub(string=strs, pattern='-+',repl='-')
  matched = re.match(string= " "+strs+" ",pattern=date_pattern,flags=re.I)
  return 1 if matched else 0

### Weekdays flag
def checkOtherDate(strs):

  days_abbr = ['saturday', 'sunday', 'monday', 'tuesday', 'wednesday','thursday', 'friday','october','november','december','january','february','march','april','may','june','july','august','september']
  days_abbr_patt = '\\b' + '\\b|\\b'.join(days_abbr) + '\\b'

  day_check = re.findall(string = strs, pattern=days_abbr_patt, flags = re.I)
  return 1 if len(day_check) >0 else 0        

def checkRange(vals):
  range_patt = '(\d+)\s*[-|to]+\s*(\d+)'
  range_data = re.findall(pattern=range_patt,string=str(vals))
  if len(range_data)>0 and float(range_data[0][1])>=float(range_data[0][0]):
    return 1

def upper_char_len_in_cells(values):
    return len(re.findall('[A-Z]', values))/len(values)

def lower_char_len_in_cells(values):
    return len(re.findall('[a-z]', values))/len(values)

def get_word_length(val):
    return len(val.split(' '))

def additional_features(col_values: list, date_samples=1000):

  #print('col_values',col_values)
  
  numeric_type = []
  int_type = []
  float_type = []
  alpha_type = []
  alphanum_type = []
  others_type = []
  upper_case = []
  lower_case = []
  range_type = []
  date_type = []
  day_type = []

  mean_before_int, mean_before_float, mean_after_float, max_after_float, zero_flag,mean_uppercase, mean_lowercase = 0,0,0,0,0,0,0
  int_mean, int_variance, int_skew, int_kurtosis, int_min, int_max, int_sum, int_median = 0,0,0,-3,0,0,0,0 
  float_mean, float_variance, float_skew, float_kurtosis, float_min, float_max, float_sum, float_median = 0,0,0,-3,0,0,0,0
  alphaNumRatio, numericRatio, alphaRatio, otherRatio, dateRatio, dayRatio, intRatio, floatRatio = 0,0,0,0,0,0,0,0
  wordlen_mean, wordlen_variance, wordlen_skew, wordlen_kurtosis, wordlen_min, wordlen_max, wordlen_sum, wordlen_median = 0,0,0,0,0,0,0,0

  col_values= [values for values in col_values if len(values)>0]
  total_vals = len(col_values)

  print('Custom feature creation alphaAndNumericMatch started:',datetime.now())
  alphaNum = pseq(map(alphaAndNumericMatch, col_values), processes=core_count, partition_size=size)
  alphaNum = list(alphaNum)
  alpha_type = list(filter(lambda item: item == 'alpha', alphaNum))
  alphanum_type = list(filter(lambda item: item == 'alphanumeric', alphaNum))
  numeric_type = list(filter(lambda item: item == 'numeric', alphaNum))
  others_type = list(filter(lambda item: item == 'others', alphaNum))
  #print('alphaNum',alphaNum)
  #print('alphaNum_Set',set(alphaNum))

  print('Custom feature creation upper and lower characters started:',datetime.now())
  upper_case_ratio = pseq(map(upper_char_len_in_cells, col_values), processes=core_count, partition_size=size)
  upper_case_ratio = list(upper_case_ratio)
  #print(upper_case_ratio)
  mean_uppercase = np.mean(upper_case_ratio) if len(upper_case_ratio)>0 else 0
  #print(upper_case_ratio,mean_uppercase)

  lower_case_ratio = pseq(map(lower_char_len_in_cells, col_values), processes=core_count, partition_size=size)
  lower_case_ratio = list(lower_case_ratio)
  #print(lower_case_ratio)
  mean_lowercase = np.mean(lower_case_ratio) if len(lower_case_ratio)>0 else 0 
  #print(lower_case_ratio,mean_lowercase)

  #print('Custom feature creation checkInt started:',datetime.now())
  int_data = pseq(map(checkInt, col_values), processes=core_count, partition_size=size)
  #print('int_data',int_data)
  int_type = [int(col_values[idx]) for idx,val in enumerate(int_data) if val==1]
  int_type = list(int_type)
  #print('int_type',int_type)

  #print('Custom feature creation checkFloat started:',datetime.now())
  float_data = pseq(map(checkFloat, col_values), processes=core_count, partition_size=size)
  #print('float_data',float_data)
  float_type = [float(col_values[idx]) for idx,val in enumerate(float_data) if val==1]
  float_type = list(float_type)
  #print('float_type',float_type)

  print('Custom feature creation checkRange started:',datetime.now())
  range_data = pseq(map(checkRange, col_values), processes=core_count, partition_size=size)
  #print('range_data',range_data)
  range_type = list(filter(lambda item: item == 1, range_data))
  range_type = list(range_type)
  #print('range_type',range_type)

  print('Custom feature creation checkDate started:',datetime.now())
  sub_values = col_values[:date_samples]    
  date_data= pseq(map(checkDate, sub_values), processes=core_count, partition_size=size)
  #print('date_data',date_data)
  checkdate = np.mean(list(date_data))

  if checkdate == 1:
    date_type = [1] * len(col_values)

  print('Custom feature creation checkOtherDate started:',datetime.now())
  day_data= pseq(map(checkOtherDate, sub_values), processes=core_count, partition_size=size)
  #print('day_data',day_data)
  checkday = np.mean(list(day_data))

  if checkday==1:
    day_type = [1] * len(col_values)

  alphaNumRatio = len(alphanum_type)/total_vals if total_vals >0 else 0
  numericRatio = len(numeric_type)/total_vals if total_vals >0 else 0
  alphaRatio = len(alpha_type)/total_vals if total_vals >0 else 0
  otherRatio = len(others_type)/total_vals if total_vals >0 else 0
  dateRatio = len(date_type)/total_vals if total_vals >0 else 0
  dayRatio = len(day_type)/total_vals if total_vals >0 else 0
  intRatio = len(int_type)/total_vals if total_vals >0 else 0
  floatRatio = len(float_type)/total_vals if total_vals >0 else 0
  rangeRatio = len(range_type)/total_vals if total_vals >0 else 0

  print('Custom feature creation Integer features:',datetime.now())
  ### Integer type
  int_median = np.median(int_type)
  int_median = int_median if pd.notnull(int_median) else 0
    
  try:
    mean_before_int = np.mean([len(str(val)) for val in int_type])
  except Exception as e:
    mean_before_int = np.NaN
    print(e)

  mean_before_int = mean_before_int if pd.notnull(mean_before_int) else 0
  
  if len(int_type)>0:
    int_mean, int_variance, int_skew, int_kurtosis, int_min, int_max, int_sum = compute_stats(int_type)

  int_mean = int_mean if pd.notnull(int_mean) else 0
  int_variance = int_variance if pd.notnull(int_variance) else 0
  int_skew = int_skew if pd.notnull(int_skew) else 0
  int_kurtosis = int_kurtosis if pd.notnull(int_kurtosis) else -3
  int_min = int_min if pd.notnull(int_min) else 0
  int_max = int_max if pd.notnull(int_max) else 0
  int_sum = int_sum if pd.notnull(int_sum) else 0     

  print('Custom feature creation Float features:',datetime.now())

  ### Float type
  float_median = np.median(float_type)
  float_median = float_median if pd.notnull(float_median) else 0
  #print('float_median',float_median)

  try: 
    float_elem = [str(float(val)).split('.') for val in float_type if pd.notnull(val)]
    mean_before_float = np.mean([len(val[0]) for val in float_elem])
    #print('mean_before_float',mean_before_float)
    mean_after_float = np.mean([len(val[1]) for val in float_elem if len(val)>1])
    #print('mean_after_float',mean_after_float)
    max_after_float = max([float(val[1]) for val in float_elem if len(val)>1])
    #print('max_after_float',max_after_float)

  except Exception as e:
    max_after_float = np.NaN
    mean_before_float, mean_after_float = 0,0
    print(e)
    
  orig_max_after_float = max_after_float
  max_after_float = max_after_float if pd.notnull(max_after_float) else 0
  #print('max_after_float2',max_after_float)
  #print('orig_max_after_float',orig_max_after_float)  

  if pd.isnull(orig_max_after_float):
    zero_flag = 0
  elif max_after_float >0:
    zero_flag = 0
  else:
    zero_flag = 1
  
  #print('zero_flag',zero_flag)

  if len(float_type)>0:
    float_mean, float_variance, float_skew, float_kurtosis, float_min, float_max, float_sum = compute_stats(float_type)
    
  float_mean = float_mean if pd.notnull(float_mean) else 0
  float_variance = float_variance if pd.notnull(float_variance) else 0
  float_skew = float_skew if pd.notnull(float_skew) else 0
  float_kurtosis = float_kurtosis if pd.notnull(float_kurtosis) else -3
  float_min = float_min if pd.notnull(float_min) else 0
  float_max = float_max if pd.notnull(float_max) else 0
  float_sum = float_sum if pd.notnull(float_sum) else 0     

  print('Custom feature creation word length features:',datetime.now())
  word_len_data = pseq(map(get_word_length, col_values), processes=core_count, partition_size=size)
  word_len_data = list(word_len_data)

  if len(word_len_data)>0:
    wordlen_mean, wordlen_variance, wordlen_skew, wordlen_kurtosis, wordlen_min, wordlen_max, wordlen_sum = compute_stats(word_len_data)
    wordlen_median = np.median(word_len_data)
    
  print('Custom feature creation case based features:',datetime.now())

  return [total_vals,alphaNumRatio, numericRatio, alphaRatio, otherRatio, dateRatio, dayRatio, intRatio, floatRatio,rangeRatio,
        int_mean, int_variance, int_skew, int_kurtosis, int_min, int_max, int_sum, int_median, mean_before_int, float_mean, float_variance, 
        float_skew, float_kurtosis, float_min, float_max, float_sum, float_median, zero_flag, mean_before_float, mean_after_float,
        max_after_float,mean_uppercase,mean_lowercase,wordlen_mean, wordlen_variance, wordlen_skew, wordlen_kurtosis, wordlen_min, 
        wordlen_max, wordlen_sum,wordlen_median]


def extract_addl_feats(col_values: list, features: OrderedDict):
  
  feats_name = ['total_vals','alphaNumRatio', 'numericRatio', 'alphaRatio', 'otherRatio', 'dateRatio', 'dayRatio', 'intRatio', 'floatRatio',
               'rangeRatio','int_mean', 'int_variance', 'int_skew', 'int_kurtosis', 'int_min', 'int_max', 'int_sum', 'int_median', 'mean_before_int',
               'float_mean', 'float_variance', 'float_skew', 'float_kurtosis', 'float_min', 'float_max', 'float_sum', 'float_median', 'zero_flag', 
               'mean_before_float', 'mean_after_float','max_after_float','mean_uppercase','mean_lowercase', 'wordlen_mean', 'wordlen_variance', 
               'wordlen_skew', 'wordlen_kurtosis', 'wordlen_min', 'wordlen_max', 'wordlen_sum','wordlen_median']
  
  start_time = datetime.now()
  print('Custom feature creation started:',start_time) 
  feats_list = additional_features(col_values)
  end_time = datetime.now()
  print('Custom feature creation completed:',end_time)  
  print('Total time taken:', end_time-start_time)
  
  for iters,name in enumerate(feats_name):
    features[name] = feats_list[iters]