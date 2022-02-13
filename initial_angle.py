import picar_4wd as fc
import time
fc.get_distance_at(-90)
time.sleep(0.5)
for i in range(-90,91,18):
    fc.get_distance_at(i)
    #time.sleep(0.1)
