#!/usr/bin/env python3

from argparse import ArgumentError
import logging

import trio

async def client(cmdStr):
    cmdStr = cmdStr + '\n'
    client_stream = await trio.open_tcp_stream("illuminati", 6563)
    async with client_stream:
        await client_stream.send_all(cmdStr.encode('latin-1'))
        async for data in client_stream:
            print(data.decode('latin-1').strip())
            return

def main(argv=None):
    if isinstance(argv, str):
        import shlex
        argv = shlex.split(argv)

    import argparse
    parser = argparse.ArgumentParser('n8led')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--off', action='store_true', help='turn all lamps off')
    group.add_argument('--on', nargs=2, type=int, help='turn one lamp on at the given level')
    group.add_argument('--status', action='store_true', help='query lamp status')

    opts = parser.parse_args(argv)

    if not opts.off and opts.on is None:
        opts.status = True

    if opts.status:
        cmdStr = 'status'
    elif opts.off:
        cmdStr = 'off'
    elif opts.on is not None:
        led, ledPower = opts.on
        if led < 1 or led > 4:
            parser.error("led must be 1..4")
        if ledPower < 0 or ledPower > 1023:
            parser.error("ledPower must be 0..1023")
        if ledPower == 0:
            cmdStr = 'off'
        else:
            cmdStr = 'on %d %d' % (led, ledPower)
    else:
        parser.error("unhandled argparse something or other ")

    print(cmdStr, opts)
    trio.run(client, cmdStr)

if __name__ == "__main__":
    main()
