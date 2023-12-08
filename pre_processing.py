import pandas as pd

def pre_processing(df, panel):
    """
    Function to perform necessary pre-processing steps on MEPS data before being used in analysis.

    necessary operations include:
        1.Create a new column, RACE that is 'White' if RACEV2X = 1 and HISPANX = 2 i.e. non Hispanic White
          and 'non-White' otherwise
        2. RENAME all columns that are PANEL/ROUND SPECIFIC
        3. Drop rows based on certain values of individual features that correspond to missing/unknown - generally < -1
        4. Compute UTILIZATION, binarize it to 0 (< 10) and 1 (>= 10)
        5. filter for useful features (as dictated by the demo)

    Args:
        df (dataframe): pandas dataframe of MEPS data
        panel (int): numerical representation for the panel of interest (MEPS data is separated into panels)

    Returns:
        a copy of the original dataframe with the described operations applied
    """
    # features to retain
    features_to_keep=['REGION','AGE','SEX','RACE','MARRY',
                                 'FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX',
                                 'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                                 'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                                 'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42', 'ADSMOK42',
                                 'PCS42','MCS42','K6SUM42','PHQ242','EMPST','POVCAT','INSCOV','UTILIZATION', 'PERWT15F']
    
    # added to avoid modifying the underlying dataset
    df = df.copy()
    df = df[df['PANEL'] == panel]
    
    def race(row):
        """
        Helper function for operation 1 from the described operations. Encodes race from numerical to
        binary white vs non-white.

        Args:
            row(key-worded iterable): key-worded iterable with race related columns and numerical encoding
            
        Returns:
            binarized encoding according to their preceding race columns.
        """
        if ((row['HISPANX'] == 2) and (row['RACEV2X'] == 1)):  
            return 'White'
        return 'Non-White'

    
    def utilization(row):
        """
        Helper function to calculate the utilization factor for an entry. Utilization can be determined
        by the sum of 5 columns.

        Args:
            row(key-worded iterable): key-worded iterable with utilization related columns.

        Returns:
            the sum of utilization related columns.
        """
        return row['OBTOTV15'] + row['OPTOTV15'] + row['ERTOT15'] + row['IPNGTD15'] + row['HHTOTD15']

    # apply race remapping
    df['RACEV2X'] = df.apply(lambda row: race(row), axis=1)
    df = df.rename(columns = {'RACEV2X' : 'RACE'})

    # apply utilization remapping
    df['TOTEXP15'] = df.apply(lambda row: utilization(row), axis=1)

    # binarize utilization according to threshold of 10
    lessE = df['TOTEXP15'] < 10.0
    df.loc[lessE,'TOTEXP15'] = 0.0
    moreE = df['TOTEXP15'] >= 10.0
    df.loc[moreE,'TOTEXP15'] = 1.0
    df = df.rename(columns = {'TOTEXP15' : 'UTILIZATION'})

    # rename columns
    df = df.rename(columns = {'FTSTU53X' : 'FTSTU', 'ACTDTY53' : 'ACTDTY', 'HONRDC53' : 'HONRDC', 'RTHLTH53' : 'RTHLTH',
                              'MNHLTH53' : 'MNHLTH', 'CHBRON53' : 'CHBRON', 'JTPAIN53' : 'JTPAIN', 'PREGNT53' : 'PREGNT',
                              'WLKLIM53' : 'WLKLIM', 'ACTLIM53' : 'ACTLIM', 'SOCLIM53' : 'SOCLIM', 'COGLIM53' : 'COGLIM',
                              'EMPST53' : 'EMPST', 'REGION53' : 'REGION', 'MARRY53X' : 'MARRY', 'AGE53X' : 'AGE',
                              'POVCAT15' : 'POVCAT', 'INSCOV15' : 'INSCOV'})

    # drop rows correlating to missingness
    df = df[df['REGION'] >= 0] # remove values -1
    df = df[df['AGE'] >= 0] # remove values -1
    df = df[df['MARRY'] >= 0] # remove values -1, -7, -8, -9
    df = df[df['ASTHDX'] >= 0] # remove values -1, -7, -8, -9

    #for all other categorical features, remove values < -1 -> encoding reasons
    df = df[(df[['FTSTU','ACTDTY','HONRDC','RTHLTH','MNHLTH','HIBPDX','CHDDX','ANGIDX','EDUCYR','HIDEG',
                             'MIDX','OHRTDX','STRKDX','EMPHDX','CHBRON','CHOLDX','CANCERDX','DIABDX',
                             'JTPAIN','ARTHDX','ARTHTYPE','ASTHDX','ADHDADDX','PREGNT','WLKLIM',
                             'ACTLIM','SOCLIM','COGLIM','DFHEAR42','DFSEE42','ADSMOK42',
                             'PHQ242','EMPST','POVCAT','INSCOV']] >= -1).all(1)]  
    
    return df[features_to_keep]