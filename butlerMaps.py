import ruamel_yaml

yaml = ruamel_yaml.YAML(typ='safe')

configMap = dict()
dataMap = dict()

nirLabReduxRoot = "/data/pfsx/redux"
nirLabConfigRoot = "/data/pfsx/config"

dataMap['spsFile'] = dict(template="pfsx/{pfsDay}/sps/PF{site}A{visit:06d}{spectrograph}{armNum}.fits")
dataMap['rampFile'] = dict(template="pfsx/{pfsDay}/sps/PF{site}B{visit:06d}{spectrograph}{armNum}.fits")

dataMap['mask'] = dict(template="{nirLabReduxRoot}/{pfsDay}/{cam}/mask-{visit:06d}{spectrograph}{armNum}.fits",
                       loaderModule='mask')
dataMap['dark'] = dict(template="{nirLabReduxRoot}/{pfsDay}/{cam}/dark-{visit:06d}{spectrograph}{armNum}.fits",
                       loaderModule='darkCube')

dataMap['isr'] = dict(template="{nirLabReduxRoot}/{pfsDay}/{cam}/isr-{visit:06d}{spectrograph}{armNum}.fits")
