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
    for i in range(11):
        time.sleep(0.05)
    speedd.deinit()
    fc.stop()

# 50mm*1grid


def forward_grid():
    speed4 = Speed(30)
    speed4.start()
    fc.forward(100)
    for i in range(1):
        time.sleep(0.1)
    speed4.deinit()
    fc.stop()


def backward_grid():
    speed4 = Speed(30)
    speed4.start()
    fc.backward(100)
    for i in range(1):
        time.sleep(0.1)
    speed4.deinit()
    fc.stop()

if __name__ == "__main__":
    turn_right_deg()
    fc.stop()

