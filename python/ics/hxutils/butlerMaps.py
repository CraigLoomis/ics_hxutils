import pandas as pd
import ruamel_yaml

yaml = ruamel_yaml.YAML(typ='safe')

configMap = dict()
dataMap = dict()

def loadPandasCsv(file):
    return pd.read_csv(file)
def loadPandasText(file):
    return pd.read_csv(file, sep='\s+')

# "root"s, but still relative to dataRoot, etc.
configKeys = dict(nirLabReduxRoot="redux",
                  nirLabConfigRoot = "config")

configMap['detector'] = dict(template="{nirLabConfigRoot}/{cam}/detector.yaml",
                             loader=yaml.load)

dataMap['spsFile'] = dict(template="pfsx/{pfsDay}/sps/PF{site}A{visit:06d}{spectrograph}{armNum}.fits")
dataMap['rampFile'] = dict(template="ramps/{pfsDay}/PF{site}B{visit:06d}{spectrograph}{armNum}.fits",
                           loaderModule='ics.hxutils.hxramp')

dataMap['reduxDir'] = dict(template="{nirLabReduxRoot}/{cam}/{pfsDay}/{experimentName}")

dataMap['mask'] = dict(template="{nirLabReduxRoot}/{cam}/calibs/mask-{visit:06d}-{cam}.fits",
                       loaderModule='mask')
dataMap['dark'] = dict(template="{nirLabReduxRoot}/{cam}/calibs/dark-{visit:06d}-{cam}.fits",
                       loaderModule='darkCube')

dataMap['isr'] = dict(template="{nirLabReduxRoot}/{cam}/{pfsDay}/{experimentName}/isr-{visit:06d}-{cam}.fits")
dataMap['postageStamp'] = dict(template="{nirLabReduxRoot}/{cam}/{pfsDay}/{experimentName}/"
                               "ps-{visit:06d}-{cam}.fits")
dataMap['dither'] = dict(template="{nirLabReduxRoot}/{cam}/{pfsDay}/{experimentName}/"
                         "dither-{wavelength:04.0f}_{row:04.0f}_{focus:03.0f}_{visit:06d}-{cam}.fits")
dataMap['rawDither'] = dict(template="{nirLabReduxRoot}/{cam}/{pfsDay}/{experimentName}/"
                         "rawDither-{wavelength:04.0f}_{row:04.0f}_{focus:03.0f}_{visit:06d}-{cam}.fits")
dataMap['ditherSpot'] = dict(template="{nirLabReduxRoot}/{cam}/{pfsDay}/{experimentName}/"
                            "ditherSpot-{int(wavelength):04d}_{int(row):04d}_{int(focus):03d}_{xstep:03d}_{ystep:03d}_{visit:06d}-{cam}.fits")

dataMap['repeatsGrid'] = dict(template="{nirLabReduxRoot}/{cam}/{pfsDay}/{experimentName}/"
                              "repeatsGrid-{wavelength:04.0f}_{row:04.0f}-{cam}.fits")
dataMap['waveGrid'] = dict(template="{nirLabReduxRoot}/{cam}/{pfsDay}/{experimentName}/"
                           "waveGrid-{wave:04.0f}-{cam}.fits")
dataMap['focusGrid'] = dict(template="{nirLabReduxRoot}/{cam}/{pfsDay}/{experimentName}/"
                            "focusGrid-{focus:03.0f}-{cam}.fits")

dataMap['rawMeasures'] = dict(template="{nirLabReduxRoot}/{cam}/{pfsDay}/{experimentName}/"
                              "rawMeasures-{visit:06d}-{cam}.txt")
dataMap['measures'] = dict(template="{nirLabReduxRoot}/{cam}/{pfsDay}/{experimentName}/"
                           "measures-{visit:06d}-{cam}.txt")
dataMap['ditherMeasures'] = dict(template="{nirLabReduxRoot}/{cam}/{pfsDay}/{experimentName}/"
                               "ditherMeasures-{visit:06d}-{cam}.txt",
                               loader=loadPandasText)
dataMap['thermalData'] = dict(template="{nirLabReduxRoot}/{cam}/{experimentName}/"
                               "thermalData-{visit:06d}-{cam}.txt",
                               loader=loadPandasCsv)
