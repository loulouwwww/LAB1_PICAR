#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from picar_4wd.pwm import PWM
from picar_4wd.adc import ADC
from picar_4wd.pin import Pin
from picar_4wd.motor import Motor
from picar_4wd.servo import Servo
from picar_4wd.ultrasonic import Ultrasonic
from picar_4wd.speed import Speed
from picar_4wd.filedb import FileDB
from picar_4wd.utils import *
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

def test3():
    speed4 = Speed(25)
    speed4.start()
    # time.sleep(2)
    fc.forward(100)
    x = 0
    for i in range(1):
        time.sleep(0.1)
        speed = speed4()
        x += speed * 0.1
        print("%smm/s"%speed)
    print("%smm"%x)
    speed4.deinit()
    fc.stop()