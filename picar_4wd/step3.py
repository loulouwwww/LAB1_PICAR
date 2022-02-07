import picar_4wd as fc
from random import randint

speed = 30


def main():
    thr = 10
    while True:
        dist = fc.get_distance_at(0)
        i = randint(1, 2)
        if (dist >= 0 and dist <= thr):
            delay = 30
            delay2 = 50
            delay3 = 225
            while(delay):
                fc.stop()
                delay-=1
            while(delay2):
                fc.backward(speed/10)
                delay2-=1
                
            if i == 1:
                while(delay3):
                    fc.turn_left(speed)
                    delay3-=1
            else:
                while(delay3):
                    fc.turn_right(speed)
                    delay3-=1
        
        fc.forward(speed)


if __name__ == "__main__":
    try:
        main()
    finally:
        fc.stop()
