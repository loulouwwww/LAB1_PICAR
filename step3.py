import picar_4wd as fc
from random import randint
import helper_functions as hf
import time
speed = 30


def main():
    thr = 10
    while True:
        scan_list = fc.scan_step(35)
        if not scan_list:
            continue

        tmp = scan_list[3:7]
        print(tmp)
        i = randint(1, 2)
        if tmp != [2,2,2,2]:
            fc.stop()
            time.sleep(0.3)
            fc.backward(speed/10)
            time.sleep(0.3)
                
            if i == 1:
               hf.turn_left_deg()
               time.sleep(1)
               
            else:
                hf.turn_right_deg()
                time.sleep(1)
                
        
        fc.forward(speed)


if __name__ == "__main__":
    try:
        main()
    finally:
        fc.stop()
