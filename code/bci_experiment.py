from numpy import int8
from pylsl import StreamInfo, StreamOutlet, local_clock
from icecream import ic
import time
import winsound
import asyncio
import random

#
# 0 = countdown
# 1 = Move your arms around
# 2 = Talk with yourself
#


def main():
    # LSL
    srate = 280
    name = 'Trigger'
    type = 'experiment'
    info = StreamInfo(name, type, 1, srate, 'float32', 'myuid34234')

    # next make an outlet
    outlet = StreamOutlet(info)

    ID = 200
    current_stim = 1
    n_samples = 50

    ic("Input ID:")
    ID = float(input())


def arithmetic_rotation():
    """
    Arithmetic: Two numbers subracted from a large number
    Mental rotation: 
    """

    for index in range(0, n_samples * len(instructions)):

        instructions = ['Calculate ' + str(random.randrange(400, 1000)) + " - " + str(random.randrange(100, 140)) + " - " + str(random.randrange(100, 140)),
                        'Rotate a box ' + str(random.randrange(2, 9)) + " times to the " + ("right" if random.randrange(2) == 0 else "left") + " and " + str(random.randrange(6, 9)) + " times " + ("up" if random.randrange(2) == 0 else "down") + "."]
        if random.randrange(0, 2) == 0:
            current = 1
        else:
            current = 2
        SendStim(0, outlet)
        ic(instructions[current-1])
        time.sleep(random.uniform(0.1, 2.1))
        ic(3)
        time.sleep(1)
        ic(2)
        time.sleep(1)
        ic(1)
        time.sleep(1)
        SendStim(current, outlet)
        ic("Now")
        winsound.Beep(1500, 250)
        time.sleep(10)
        winsound.Beep(1000, 250)


def SendStim(stim, outlet):
    outlet.push_sample([stim])


if __name__ == '__main__':
    main()
