#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from picar_4wd.speed import Speed
import picar_4wd as fc
import time
import math


def turn_right_deg(w=13, l=13, deg=90):
    print('right'+str(deg))
    d = (w**2+l**2)**0.5*math.pi*deg/360
    print(d)
    a = int(d)
    speedd = Speed(a)
    speedd.start()
    fc.turn_right(100)
    x = 0
    for i in range(11):
        time.sleep(0.05)
        speed = speedd()
        x += speed * 0.1
        print("%smm/s" % speed)
    print("%smm" % x)
    speedd.deinit()
    fc.stop()


def turn_left_deg(w=13, l=13, deg=90):
    print('left'+str(deg))
    d = (w**2+l**2)**0.5*math.pi*deg/360
    print(d)
    a = int(d)
    speedd = Speed(a)
    speedd.start()
    fc.turn_left(100)
    x = 0
    for i in range(1):
        time.sleep(0.1)
        speed = speedd()
        x += speed * 0.1
        print("%smm/s" % speed)
    print("%smm" % x)
    speedd.deinit()
    fc.stop()

# 50mm*1grid


def forward_grid(l=50, g=1):
    speed4 = Speed(l*g)
    speed4.start()
    fc.forward(100)
    x = 0
    for i in range(1):
        time.sleep(0.1)
        speed = speed4()
        x += speed * 0.1
    #     print("%smm/s"%speed)
    # print("%smm"%x)
    speed4.deinit()
    fc.stop()


def backward_grid(l=50, g=1):
    speed4 = Speed(l*g)
    speed4.start()
    fc.backward(100)
    x = 0
    for i in range(1):
        time.sleep(0.1)
        speed = speed4()
        x += speed * 0.1
    #     print("%smm/s"%speed)
    # print("%smm"%x)
    speed4.deinit()
    fc.stop()
