import numpy as np
import pandas as pd

def filter_61722(df, startVisit=0):
    dfOut = df[df.visit >= startVisit].copy()
    dfOut = dfOut[dfOut.testName != "TEST"]
    dfOut = dfOut[dfOut.testName != "cplTest"]
    dfOut = dfOut.copy().reindex()

    dfOut.loc[df.medianFlux > 0.1, 'medianFlux'] = np.nan

    # Earliest idg.temps channels not periodically sampled.
    dfOut.loc[df.visit <= 67135,
              ['bodyTemp1', 'bodyTemp2', 'bodyTemp3', 'bodyTemp4',
               'roomTemp1', 'roomTemp2']] = np.nan

    dfOut.loc[dfOut.bodyTemp1 < 0,
              ['bodyTemp1', 'bodyTemp2', 'bodyTemp3', 'bodyTemp4',
               'roomTemp1', 'roomTemp2']] = np.nan

    # Preliminary tests could be grouped.
    dfOut.loc[dfOut.testName == 'testNotDark', 'testName'] = 'notDark0_103.5K'
    dfOut.loc[dfOut.testName == 'testNotDark_105.5K', 'testName'] = 'notDark_105.5K'
    dfOut.loc[dfOut.testName == 'testNotDark_107.5K', 'testName'] = 'notDark_107.5K'
    dfOut.loc[dfOut.testName == 'testNotDark_103.5K', 'testName'] = 'notDark_103.5K'
    dfOut.loc[dfOut.testName == 'later2_103.5K', 'testName'] = 'more_103.5K'
    dfOut.loc[dfOut.testName == 'later3_103.5K', 'testName'] = 'more_103.5K'
    dfOut.loc[dfOut.testName == 'later4_103.5K', 'testName'] = 'more_103.5K'

    # Note plate cooling test
    dfOut.loc[dfOut.testName == 'last_103.5K', 'testName'] = 'plate_test'
    dfOut.loc[dfOut.testName == 'last2_103.5K', 'testName'] = 'post_plate_test'

    # Once stable, we ran a burnoff, but repeating previous temp. steps.
    dfOut.loc[dfOut.testName == 'cooled_105.5K', 'testName'] = 'burnoff1_105.5K'
    dfOut.loc[dfOut.testName == 'cooled_107.5K', 'testName'] = 'burnoff1_107.5K'
    dfOut.loc[dfOut.testName == 'cooled_108.5K', 'testName'] = 'burnoff1_108.5K'
    dfOut.loc[(dfOut.testName == 'cooled_103.5K') & (dfOut.visit > 67297), 'testName'] = 'check_103.5K'

    # Botched first H4 change to 100K
    dfOut.loc[(dfOut.testName == 'cooled_100K') & (dfOut.visit < 67334), 'testName'] = 'check_103.5K'
    dfOut.loc[dfOut.testName == 'cooled_100K_2', 'testName'] = 'cooled_100K_0'

    # Ignore readings when H4 temp changing.
    dfOut.loc[dfOut.testName == 'slewing_103.5K', 'useForTest'] = False
    dfOut.loc[dfOut.testName == 'slewing_100K', 'useForTest'] = False
    dfOut.loc[dfOut.testName == 'slewing_100K_2', 'useForTest'] = False

    # Renames for clarity
    dfOut.loc[dfOut.testName == 'leakTest', 'testName'] = 'lightLeakTest'

    return dfOut

filterFuncs = {67122:filter_61722}

def filterSamples(df, visit0, startVisit=0):
    func = filterFuncs[visit0]
    return func(df, startVisit=startVisit)
