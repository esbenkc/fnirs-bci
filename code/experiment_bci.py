from numpy import int8
from pylsl import StreamInfo, StreamOutlet, local_clock
from icecream import ic
import time
import winsound
import asyncio
import random
import getopt
import sys
import bcolors


def main(tasks):
    """
    Experimental task.
    tasks: Pair of tasks to run.
    """
    if tasks not in ["arithmetic_audiobook", "arithmetic_rotation", "arms_talk"]:
        raise ValueError(
            "Tasks must be one of: arithmetic_audiobook, arms_talk and arithmetic_rotation")
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
    instructions = 2

    print(f"\n{bcolors.HELP}Started LSL output stream called '{name}' with type '{type}' that returns 0, 1 or 2 based on waiting period, task 1 and task 2 for {n_samples} trials each of {instructions} instructions with the tasks {tasks}.{bcolors.ENDC}")

    print(f"{bcolors.HEADER}\nStarting experiment...{bcolors.ENDC}")
    print(f"{bcolors.HELP}(Cancel with CTRL/CMD + C or go through all {n_samples*instructions} trials){bcolors.ENDC}\n")
    ic("Input the participant ID:")
    ID = float(input())

    for index in range(0, n_samples * instructions):
        if tasks == "arithmetic_audiobook":
            current = arithmetic_audiobook(outlet)
        elif tasks == "arithmetic_rotation":
            current = arithmetic_rotation(outlet)
        elif tasks == "arms_talk":
            current = arms_talk(outlet)
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


def arithmetic_rotation(outlet):
    """
    Arithmetic: Two numbers subracted from a large number
    Mental rotation: Imagining rotating a cube
    """
    instructions = ['Calculate ' + str(random.randrange(400, 1000)) + " - " + str(random.randrange(100, 140)) + " - " + str(random.randrange(100, 140)) + ".",
                    'Rotate a box ' + str(random.randrange(2, 9)) + " times to the " + ("right" if random.randrange(2) == 0 else "left") + " and " + str(random.randrange(6, 9)) + " times " + ("up" if random.randrange(2) == 0 else "down") + "."]
    if random.randrange(0, 2) == 0:
        current = 1
    else:
        current = 2
    SendStim(0, outlet)
    ic(instructions[current-1])
    return current


def arithmetic_audiobook(outlet):
    """
    Arithmetic: Two numbers subracted from a large number
    Audiobook: Listening to audiobook
    """
    instructions = ['Calculate ' + str(random.randrange(600, 900)) + " - " + str(random.randrange(20, 50)) + " - " + str(random.randrange(20, 50)) + ".",
                    "Listen to audiobook."]
    if random.randrange(0, 2) == 0:
        current = 1
    else:
        current = 2
    SendStim(0, outlet)
    ic(instructions[current-1])
    return current


def arms_talk(outlet):
    """
    Arms: Move both arms
    Talk: Talk with yourself about anything
    """
    instructions = ['Move both arms.',
                    'Talk with yourself.']
    if random.randrange(0, 2) == 0:
        current = 1
    else:
        current = 2
    SendStim(0, outlet)
    ic(instructions[current-1])
    return current


def SendStim(stim, outlet):
    outlet.push_sample([stim])


if __name__ == '__main__':
    args = getopt.getopt(sys.argv[1:], "hi:o:", ["ifile=", "ofile="])
    try:
        i = args[1][0]
    except Exception as e:
        raise ValueError(f"{bcolors.FAIL}\n\nAdd a task to run by writing it when calling the experiment:\n\n1) arithmetic_audiobook\n2) arithmetic_rotation\n3) arms_talk\n\nI.e. python3 bci_experiment.py arithmetic_audiobook\n{bcolors.ENDC}")
    main(i)
