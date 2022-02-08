import functools
import itertools
import logging

import gpiozero
import trio

logging.basicConfig(format='%(asctime)s %(name)s  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

class GpioBinarySelect(object):
    def __init__(self, pins=None, values=None):
        self.logger = logging.getLogger('select')
        self.bits = []
        self.map = values
        self.invmap = {v:k for k,v in self.map.items()}
        for p in pins:
            self.bits.append(gpiozero.OutputDevice(p))

    def __del__(self):
        for p in self.bits:
            p.close()
            
    def __str__(self):
        return "BinarySelect(%d,%d)" % (self.bits[0].value,
                                        self.bits[1].value)

    def status(self):
        bitMask = (self.bits[0].value == 1, self.bits[1].value == 1)
        return self.invmap[bitMask]
    
    def setLed(self, led):
        p1, p2 = self.map[led]
        self.logger.info('setting lamp %s, bits=%d,%d', led, p1, p2)
        self.bits[0].value = p1
        self.bits[1].value = p2
        
class GpioLadder(object):
    def __init__(self, pins=None):
        self.logger = logging.getLogger('ladder')
        self.bits = []
        for p in pins:
            self.bits.append(gpiozero.OutputDevice(p))

    def __del__(self):
        for p in self.bits:
            p.close()
            
    def __str__(self):
        val = self.status()
        return "GpioLadder(value=%d, 0x%02x)" % (val, val)

    def status(self):
        val = 0
        for b_i, b in enumerate(self.bits[::1]):
            val += (1<<b_i) if b.value else 0

        return val
        
    def setLevel(self, value):
        if not isinstance(value, int) or value < 0 or (value > 2**(len(self.bits))-1):
            raise ValueError('invalid level: %s' % (value))

        self.logger.info('setting level=%d', value)
        for i in range(len(self.bits)):
            doSet = (value & 1<<i) > 0
            self.logger.debug('setting %s bit to %d', self.bits[i], value)
            self.bits[i].value = doSet
            
            
class ColdIlluminator(object):
    """Control the 4-LED controller on the n8 cryostat.

    GPIO2 - LED power on
    GPIO3,4,14,15,17,18,27,22,23,24 - bits 0..7 of the resistor ladder
    GPIO26,20 - select single active LED from four.

    """

    def __init__(self, logLevel=logging.INFO):
        self.logger = logging.getLogger('illuminator')
        self.logger.setLevel(level=logLevel)

        self.powerAll = gpiozero.OutputDevice(pin='GPIO2')
        self.ledSelect = GpioBinarySelect(pins=['GPIO26', 'GPIO20'], 
                                          values={1:(True, False),
                                                  2:(False, False),
                                                  3:(True, True),
                                                  4:(False, True)})
        self.ledPower = GpioLadder(['GPIO3', 'GPIO4',
                                    'GPIO14', 'GPIO15',
                                    'GPIO17', 'GPIO18',
                                    'GPIO27', 'GPIO22',
                                    'GPIO23', 'GPIO24'])
        
        self.led = None

    def __del__(self):
        self.powerAll.close()
            
    def __str__(self):
        return "ColdIlluminator(power=%s, led=%s, level=%s)" % (self.powerAll.value,
                                                                self.ledSelect,
                                                                self.ledPower)

    def status(self):
        return (int(self.powerAll.value == 1),
                self.ledSelect.status(),
                self.ledPower.status())
                
    def setLED(self, ledNum, level):
        if ledNum == 0 or level == 0:
            self.powerAll.off()
            self.ledSelect.setLed(2)
            self.ledPower.setLevel(0)
        elif ledNum >= 1 and ledNum <= 4:
            self.ledSelect.setLed(ledNum)
            self.ledPower.setLevel(level)
            self.powerAll.on()
        else:
            raise ValueError('ledNum must be 0 or 1..4, not %s' % (ledNum))

        return self.status()

async def reply(stream, status, text=''):
    reply = "{} {}\n".format(status, text).encode('latin-1')
    await stream.send_all(reply)
    
async def cmdServer(stream, illuminator=None, connectionIds=None):
    logger = logging.getLogger('ledServer')
    logger.setLevel(logging.INFO)
    thisId =  next(connectionIds)
    
    logger.info("led_server {} started".format(thisId))
    try:
        async for data in stream:
            logger.info("led_server {}: received data {!r}".format(thisId, data))
            try:
                data = data.decode('latin-1').strip()
                if data == 'off':
                    ledNum = None
                    ledLevel = 0
                elif data == 'status':
                    ret = illuminator.status()
                    await reply(stream, 'OK', ' '.join([str(x) for x in ret]))
                    continue            
                elif data.startswith('on'):
                    parts = data.split()
                    if len(parts) != 3:
                        await reply(stream, 'ERROR', 'on ledNum ledLevel')
                        continue
                    try:
                        ledNum = int(parts[1], base=10)
                        ledLevel = int(parts[2], base=10)
                    except ValueError as e:
                        await reply(stream, 'ERROR', e)
                        continue
                else:
                    await reply(stream, 'ERROR', 'unknown command: {}'.format(data))
                    continue
            except Exception as e:
                await reply(stream, 'ERROR', e)
                continue
                
            ret = illuminator.setLED(ledNum, ledLevel)
            await reply(stream, 'OK', ' '.join([str(x) for x in ret]))

        logger.info("led_server {}: connection closed".format(thisId))
    # FIXME: add discussion of MultiErrors to the tutorial, and use
    # MultiError.catch here. (Not important in this case, but important if the
    # server code uses nurseries internally.)
    except Exception as exc:
        # Unhandled exceptions will propagate into our parent and take
        # down the whole program. If the exception is KeyboardInterrupt,
        # that's what we want, but otherwise maybe not...
        logger.error("led_server {}: crashed: {!r}".format(thisId, exc))

async def main():
    connectionIds = itertools.count()
    leds = ColdIlluminator()
    server = functools.partial(cmdServer, illuminator=leds, connectionIds=connectionIds)
    await trio.serve_tcp(server, 6563)
    
if __name__ == "__main__":
    trio.run(main)
