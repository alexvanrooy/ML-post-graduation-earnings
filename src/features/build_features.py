#IMPORTS
import pandas as pd
import sys

def select_features():
    '''
    Method for separating the features from the target variable
    '''
    input_path = "..\data\interim\Most-Recent-Cohorts-Institution_CLEANED.csv"
    X_output_path = '..\data\interim\X.csv'
    y_output_path = '..\data\interim\y.csv'

    dataset = None

    try:
        dataset = pd.read_csv(input_path)
    except Exception as e:
        print("ERROR: Could not load cleaned dataset.")
        print(e)
        return 0

    #List of selected features to use
    default_features = [
        'SCH_DEG',
        'MAIN',
        'NUMBRANCH',
        'PREDDEG',
        'HIGHDEG',
        'CONTROL',
        'CCBASIC',
        'CCUGPROF',
        'CCSIZSET',
        'DISTANCEONLY',
        'UGDS',
        'UGDS_WHITE',
        'UGDS_BLACK',
        'UGDS_HISP',
        'UGDS_ASIAN',
        'UGDS_AIAN',
        'UGDS_NHPI',
        'UGDS_2MOR',
        'UGDS_NRA',
        'UGDS_UNKN',
        'PPTUG_EF',
        'TUITFTE',
        'INEXPFTE',
        'POOLYRS',
        'POOLYRS200',
        'PCTFLOAN',
        'CDR2',
        'CDR3',
        'COMP_ORIG_YR2_RT',
        'WDRAW_ORIG_YR2_RT',
        'COMP_ORIG_YR3_RT',
        'WDRAW_ORIG_YR3_RT',
        'COMP_ORIG_YR4_RT',
        'WDRAW_ORIG_YR4_RT',
        'COMP_ORIG_YR6_RT',
        'WDRAW_ORIG_YR6_RT',
        'COMP_ORIG_YR8_RT',
        'WDRAW_ORIG_YR8_RT',
        'RPY_1YR_RT',
        'COMPL_RPY_1YR_RT',
        'NONCOM_RPY_1YR_RT',
        'RPY_3YR_RT',
        'COMPL_RPY_3YR_RT',
        'NONCOM_RPY_3YR_RT',
        'RPY_5YR_RT',
        'COMPL_RPY_5YR_RT',
        'NONCOM_RPY_5YR_RT',
        'RPY_7YR_RT',
        'INC_PCT_LO',
        'INC_PCT_M1',
        'INC_PCT_M2',
        'PAR_ED_PCT_PS',
        'DEBT_MDN',
        'GRAD_DEBT_MDN',
        'WDRAW_DEBT_MDN',
        'FEMALE_DEBT_MDN',
        'MALE_DEBT_MDN',
        'DEBT_N',
        'GRAD_DEBT_N',
        'WDRAW_DEBT_N',
        'FEMALE_DEBT_N',
        'MALE_DEBT_N',
        'FAMINC',
        'MD_FAMINC',
        'FAMINC_IND',
        'PCT_WHITE',
        'PCT_BLACK',
        'PCT_ASIAN',
        'PCT_HISPANIC',
        'PCT_BA',
        'PCT_GRAD_PROF',
        'PCT_BORN_US',
        'MEDIAN_HH_INC',
        'POVERTY_RATE',
        'UNEMP_RATE',
        'ICLEVEL',
        'UGDS_MEN',
        'UGDS_WOMEN',
        'OPENADMP',
        'FTFTPCTPELL',
        'FTFTPCTFLOAN',
        'CNTOVER150_1YR'
    ]
    target = ['MN_EARN_WNE_P6']

    X = dataset[default_features]
    y = dataset[target]

    try:
        X.to_csv(X_output_path)
        y.to_csv(y_output_path)
        print("Number of features used: ", X.shape[1])
    except Exception as e:
        print('ERROR: Something went wrong.')
        print(e)
        return 0
    
    return 1