#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import picar_4wd as fc
import time


def turn_right_deg():
    fc.stop()
    time.sleep(0.2)
    fc.turn_right(100)
    for i in range(22):
        time.sleep(0.025)
    fc.stop()
    time.sleep(0.2)


def turn_left_deg():
    fc.stop()
    time.sleep(0.2)
    fc.turn_left(100)
    for i in range(19):
        time.sleep(0.025)
    fc.stop()
    time.sleep(0.2)
# 50mm*2grid

# n means grid


def forward_grid(n=1):
    fc.forward(20)
    for i in range(n):
        time.sleep(0.10)
    fc.stop()



def backward_grid(n=1):
    fc.backward(100)
    for i in range(n):
        time.sleep(0.10)
    fc.stop()


if __name__ == "__main__":
    # forward_grid(2)
    forward_grid(5)
    turn_left_deg()
    fc.stop()