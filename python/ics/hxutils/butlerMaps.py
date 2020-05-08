import ruamel_yaml

yaml = ruamel_yaml.YAML(typ='safe')

configMap = dict()
dataMap = dict()

configKeys = dict(nirLabReduxRoot="/data/redux",
                  nirLabConfigRoot = "/data/pfsx/config")

configMap['detector'] = dict(template="{nirLabConfigRoot}/{cam}/detector.yaml",
                             loader=yaml.load)

dataMap['spsFile'] = dict(template="pfsx/{pfsDay}/sps/PF{site}A{visit:06d}{spectrograph}{armNum}.fits")
dataMap['rampFile'] = dict(template="pfsx/{pfsDay}/sps/PF{site}B{visit:06d}{spectrograph}{armNum}.fits")

dataMap['reduxDir'] = dict(template="{nirLabReduxRoot}/{cam}/{pfsDay}/{experimentName}")

dataMap['mask'] = dict(template="{nirLabReduxRoot}/{cam}/calibs/mask-{visit:06d}-{cam}.fits",
                       loaderModule='ics.hxutils.mask')
dataMap['dark'] = dict(template="{nirLabReduxRoot}/{cam}/calibs/dark-{visit:06d}-{cam}.fits",
                       loaderModule='ics.hxutils.darkCube')

dataMap['isr'] = dict(template="{reduxDir}/isr-{visit:06d}-{cam}.fits")
dataMap['postageStamp'] = dict(template="{reduxDir}/ps-{visit:06d}-{cam}.fits")
dataMap['dither'] = dict(template="{reduxDir}/dither-{wave:04.0f}_{row:04.0f}_{focus:03.0f}_{visit:06d}-{cam}.fits")

dataMap['repeatsGrid'] = dict(template="{reduxDir}/repeatsGrid-{wave:04.0f}_{row:04.0f}-{cam}.fits")
dataMap['waveGrid'] = dict(template="{reduxDir}/waveGrid-{wave:04.0f}-{cam}.fits")
dataMap['focusGrid'] = dict(template="{reduxDir}/focusGrid-{focus:03.0f}-{cam}.fits")

dataMap['measures'] = dict(template="{reduxDir}/measures-{visit:06d}-{cam}.txt")

