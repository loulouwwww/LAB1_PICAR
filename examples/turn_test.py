import picar_4wd as fc
import helper_functions as hf
import time

speed = 30


def s_p_test():
    # please check the real distance in every step
    fc.forward(speed)
    time.sleep(0.2)
    fc.stop()
    time.sleep(1)
    fc.forward(150)
    time.sleep(0.2)
    fc.stop()
    time.sleep(1)
    fc.forward(2*speed)
    time.sleep(0.1)
    fc.stop()
    time.sleep(1)


def main():
    # test1 speed&power test
    #s_p_test()
    # test2 turn test
    hf.turn_right_deg(deg=90)
    time.sleep(1)
    fc.stop()
    hf.turn_right_deg(deg=90)
    time.sleep(1)
    fc.stop()
    #hf.turn_left_deg(deg=90)
    time.sleep(1)
    fc.stop()
    #hf.turn_left_deg(deg=90)
    time.sleep(1)
    fc.stop()
    #hf.turn_right_deg(deg=180)
    #time.sleep(1)
    #hf.turn_left_deg(deg=240)
    time.sleep(1)
    hf.turn_right_deg(deg=90)
    time.sleep(1)
    fc.stop()
    hf.turn_right_deg(deg=90)
    time.sleep(1)
    fc.stop()
    hf.turn_right_deg(deg=90)
    time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    finally:
        fc.stop()
