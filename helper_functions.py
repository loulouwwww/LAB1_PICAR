#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from picar_4wd.speed import Speed
import picar_4wd as fc
import time
import math


def turn_right_deg():

    fc.turn_right(100)
    for i in range(11):
        time.sleep(0.05)
    fc.stop()
    time.sleep(0.1)


def turn_left_deg():
    fc.turn_left(100)
    for i in range(10):
        time.sleep(0.05)
    fc.stop()
    time.sleep(0.1)

# 50mm*2grid

# n means grid


def forward_grid(n=1):
    fc.forward(100)
    for i in range(n):
        time.sleep(0.1)
    fc.stop()
    time.sleep(0.1)


def backward_grid(n=1):
    fc.backward(100)
    for i in range(n):
        time.sleep(0.1)
    fc.stop()
    time.sleep(0.1)


if __name__ == "__main__":
    # forward_grid(2)
    backward_grid(2)
    fc.stop()
