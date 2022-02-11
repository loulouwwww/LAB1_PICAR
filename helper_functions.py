#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from picar_4wd.speed import Speed
import picar_4wd as fc
import time
import math


def turn_right_deg():
    speedd = Speed(30)
    speedd.start()
    fc.turn_right(100)
    for i in range(11):
        time.sleep(0.05)
    speedd.deinit()
    fc.stop()


def turn_left_deg():
    speedd = Speed(30)
    speedd.start()
    fc.turn_left(100)
    for i in range(10):
        time.sleep(0.05)
    speedd.deinit()
    fc.stop()

# 50mm*2grid

# n means grid
def forward_grid(n=1):
    speed4 = Speed(30)
    speed4.start()
    fc.forward(100)
    for i in range(n):
        time.sleep(0.1)
    speed4.deinit()
    fc.stop()


def backward_grid(n=1):
    speed4 = Speed(30)
    speed4.start()
    fc.backward(100)
    for i in range(n):
        time.sleep(0.1)
    speed4.deinit()
    fc.stop()

if __name__ == "__main__":
    #forward_grid(2)
    backward_grid(2)
    fc.stop()
