import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_tcp, hooks

import logging

logger = modbus_tk.utils.create_logger("console", level=logging.DEBUG)

def on_after_recv(args):
    response = args[1]
    logger.debug("on_after_recv {0} bytes received".format(len(response)))
hooks.install_hook("modbus_tcp.TcpMaster.after_recv", on_after_recv)

class N8Reeds:
    def __init__(self, verbose=False):
        """Control the n8 LED select and power switches.

        We have switches 8-16 (0-based)
        Within that:
         - 8-11  per-led grounds
         - 12-15 per-led power outputs

        To best turn things off: open all outputs, close all grounds
        
        """
        self.verbose = verbose
        self.master = None
        self.connect()

        self.switch0 = 8
        self.ground0 = 8
        self.output0 = 12
        
        self.makeLive = 0

    def __str__(self):
        return f'N8Reeds({self.getState()}'
    
    def __del__(self):
        self.master.close()
        del self.master
        
    def connect(self):
        if self.master is not None:
            self.master.close()
            self.master = None
            
        self.master = modbus_tcp.TcpMaster('ampSwitch')
        self.master.set_verbose(255 if self.verbose else 0)

    def _getSwitches(self):
        ret = self.master.execute(1, cst.READ_COILS, 0, 16)
        return ret

    def getState(self):
        ret = self._getSwitches()
        ret = ret[8:]

        grounds = ret[:4]
        outputs = ret[4:]

        states = [grounds[i] | (outputs[i] << 1) for i in range(4)]
        return states

    def status(self):
        rawState = self.getState()
        led = 0
        for i in range(4):
            s = rawState[i]
            if s == 2:
                led = i+1
                break
        return led
    
    def _setSwitch(self, switchId, state):
        self.master.execute(1, cst.WRITE_SINGLE_COIL, switchId, 
                            output_value=int(bool(state)))
        ret = self._getSwitches()
        return ret
    
    def _setSwitches(self, states, switch0=None):

        if switch0 is None:
            switch0 = self.switch0
            
        switchStates = [int(bool(s)) for s in states]
        self.master.execute(1, cst.WRITE_MULTIPLE_COILS, 
                            switch0, 
                            output_value=switchStates)

    def _setGrounds(self, states):
        """Set the four ground switches."""
        if len(states) != 4:
            raise ValueError('need exactly four values')

        self._setSwitches(states, switch0=self.ground0)
        
    def _setOutputs(self, states):
        """Set the four output switches."""
        if len(states) != 4:
            raise ValueError('need exactly four values')

        self._setSwitches(states, switch0=self.output0)
        
    def ledsOff(self):
        """Turn all LEDs off as best we can.

        Close all ground switches, open all output switches.
        """

        self._setOutputs([0,0,0,0])
        self._setGrounds([1,1,1,1])
        return self.getState()
    
    def ledOn(self, led):
        """Turn single LED on.

        Args
        ----
        led : `int`
          Which led to enable. 1-based
        """

        if led > 4 or led < 0:
            raise ValueError(f'led must be 0 or 1..4 ({led}')
        if led == 0:
            return self.ledsOff()
        
        grounds = [1]*4
        outputs = [0]*4

        grounds[led - 1] = 0
        outputs[led - 1] = 1

        self._setGrounds(grounds)
        self._setOutputs(outputs)
        
        return self.getState()        
