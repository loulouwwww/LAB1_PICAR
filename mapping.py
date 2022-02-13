import heapq
import math
import sys
import threading
import time
from unittest import case

import cv2
import matplotlib.pyplot as plt
import numpy as np

import helper_functions as hf
import picar_4wd as fc
import utils
from object_detector import ObjectDetector, ObjectDetectorOptions

size = 20  # size of local map
unit = 5  # cm/grid
car_width = 16  # cm
car_length = 23.5
half_lg = int(car_length/unit / 2)  # 2
half_wg = int(car_width/unit/2)  # 1
half_size = int(size/2)
real_local_obs = []  # [y,x]
real_global_obs = []
multiple = 3

polar_map = []
cart_map = np.zeros((size+1, size+1), dtype=int)  # local map
total_size = size*multiple+1
global_map = np.zeros((total_size, total_size),
                      dtype=int)  # local map

init_y = int(size*multiple/4)
init_x = int(size*multiple/2)
curr_y = int(size*multiple/4)
curr_x = int(size*multiple/2)
target_y = 5*init_y
target_x = init_x
# Direction:0:screen down(0), 1:screen right(left90), 2:screen up(180), 3:screen left(right 90)
curr_dir = 0
# 0:routing 1:reach
movement_list = []
# map mark: 0:blank 1:real obs 2:fake obs 3:car center 4:car, 5:target
# assume car w: 3 grids,l: 5 grids
cv_detected = 0


def polar_mapping(step=18):
    global polar_map
    polar_map = []
    fc.get_distance_at(-90)
    time.sleep(0.5)
    for angle in range(-90, 91, step):
        angdis = [angle, fc.get_distance_at(angle)]
        print(angdis)
        polar_map.append(angdis)
    return


def polar_to_cartesian():
    global cart_map, real_local_obs
    cart_map = np.zeros((size, size+1), dtype=int)
    real_local_obs = []
    for i in polar_map:
        if i[1] >= 0 and i[1] < size*unit/2:
            x = int(i[1]*math.sin(i[0]/180*math.pi)/unit)+half_size
            y = int(i[1]*math.cos(i[0]/180*math.pi)/unit)
            real_local_obs.append([y, x])
            cart_map[y][x] = 1

    return


def bound(base_x, base_y):
    base_x = max(0, base_x)
    base_x = min(size*multiple, base_x)
    base_y = max(0, base_y)
    base_y = min(size*multiple, base_y)
    return base_x, base_y


def mark_car():
    global cart_map, real_obs, global_map, fake_obs, curr_x, curr_y, curr_status
    curr_x, curr_y = bound(curr_x, curr_y)
    print(curr_y, curr_x)
    if global_map[curr_y][curr_x] == 5:
        return
    else:
        global_map[curr_y][curr_x] = 3
    # for i in range(-half_wg, half_wg+1):
    #     for j in range(-half_lg, half_lg+1):
    #         if curr_dir % 2 == 0:
    #             y = curr_y+j
    #             x = curr_x+i
    #         else:
    #             y = curr_y+i
    #             x = curr_x+j
    #         x, y = bound(x, y)
    #         if global_map[y][x] != 3:
    #             global_map[y][x] = 4
    return

# transform the local coord to global coord


def local_to_global(local_y, local_x):
    global_y = 0
    global_x = 0
    if curr_dir == 0:
        global_y = curr_y+local_y + (half_lg+1)
        global_x = curr_x+local_x-half_size
    elif curr_dir == 1:
        global_y = curr_y-local_x+half_size
        global_x = curr_x+local_y+(half_lg+1)
    elif curr_dir == 2:
        global_y = curr_y-local_y - (half_lg+1)
        global_x = curr_x-local_x+half_size
    elif curr_dir == 3:
        global_y = curr_y+local_x - half_size
        global_x = curr_x-local_y-(half_lg+1)
    global_x, global_y = bound(global_x, global_y)
    return global_y, global_x
# transform the global coord to local coord


def global_to_local(global_y, global_x):
    local_y = 0
    local_x = 0
    if curr_dir == 0:
        local_y = global_y-curr_y-(half_lg+1)
        local_x = global_x-curr_x+half_size
    elif curr_dir == 1:
        local_x = curr_y-global_y+half_size
        local_y = global_x-curr_x-(half_lg+1)
    elif curr_dir == 2:
        local_y = curr_y-global_y - (half_lg+1)
        local_x = curr_x-global_x+half_size
    elif curr_dir == 3:
        local_x = global_y-curr_y+half_size
        local_y = curr_x-global_x-(half_lg+1)

    return local_y, local_x


def mark_obs():
    global global_map, real_global_obs
    r2 = 9
    bound_f = half_lg+1
    print('old obs', real_global_obs)
    remove_list = []
    for obs in real_global_obs:
        ly, lx = global_to_local(obs[0], obs[1])
        # print(ly, lx)
        # print(obs[0], obs[1])
        if ly >= -1 and ly <= size and lx >= -half_size-1 and lx <= half_size+1:
            remove_list.append(obs)
            global_map[obs[0], obs[1]] = 0
            # remove 2 ways
            # for i in range(-bound_f, bound_f+1):
            #     for j in range(-bound_f, bound_f + 1):
            #         if global_map[obs[0]+i][obs[1]+j] == 2:
            #             global_map[obs[0]+i][obs[1]+j] = 0
            for i in range(-bound_f, bound_f+1):
                for j in range(-bound_f, bound_f + 1):
                    # no bound check
                    o_x, o_y = bound(obs[1]+j, obs[0]+i)
                    if i**2+j**2 <= r2 and global_map[o_y][o_x] == 2:
                        global_map[o_y][o_x] = 0
    print('removed obs', remove_list)
    for obs in remove_list:
        real_global_obs.remove(obs)
    # print('add')
    for obs in real_local_obs:
        base_y, base_x = local_to_global(obs[0], obs[1])
        global_map[base_y][base_x] = 1
        # print(base_y, base_x)
        real_global_obs.append([base_y, base_x])
        # padding 2 ways
        # for i in range(-bound_f, bound_f+1):
        #     for j in range(-bound_f, bound_f + 1):
        #         if global_map[base_y+i][base_x+j] == 0:
        #             global_map[base_y+i][base_x+j] = 2
        for i in range(-bound_f, bound_f+1):
            for j in range(-bound_f, bound_f + 1):
                # no bound check
                o_x, o_y = bound(base_x+j, base_y+i)
                if i**2+j**2 <= r2 and global_map[o_y][o_x] == 0:
                    global_map[o_y][o_x] = 2
    print('new obs', real_global_obs)
    return


def update_map():
    # global cart_map, real_obs, global_map, fake_obs
    polar_mapping()
    polar_to_cartesian()
    mark_obs()
    mark_car()
    plot()
    return


def move_forward():
    global curr_y, curr_x, curr_dir
    if curr_dir == 0:
        curr_y += 1
    elif curr_dir == 1:
        curr_x += 1
    elif curr_dir == 2:
        curr_y -= 1
    else:
        curr_x -= 1
    mark_car()
    hf.forward_grid()
    return


def move_backward():
    global curr_y, curr_x
    if curr_dir == 0:
        curr_y -= 1
    elif curr_dir == 1:
        curr_x -= 1
    elif curr_dir == 2:
        curr_y += 1
    else:
        curr_x += 1
    mark_car()
    hf.backward_grid()
    return


def turn_left():
    global curr_dir, curr_y, curr_x
    curr_dir = (curr_dir+1) % 4


def move_left():
    turn_left()
    move_forward()
    mark_car()
    hf.turn_left_deg()
    hf.forward_grid()

    return


def turn_right():
    global curr_dir, curr_y, curr_x
    curr_dir = (curr_dir-1) % 4


def move_right():
    turn_right()
    move_forward()
    mark_car()
    hf.turn_right_deg()
    hf.forward_grid()
    return


def plot():
    plt.figure()
    plt.imshow(global_map)
    plt.show()


def detect():
    global cv_detected
    width = 480
    height = 320
    num_threads = 2
    model = 'efficientdet_lite0.tflite'
    camera_id = 0
    enable_edgetpu = False
    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Visualization parameters
    row_size = 20  # pixels
    left_margin = 24  # pixels
    text_color = (0, 0, 255)  # red
    font_size = 1
    font_thickness = 1
    fps_avg_frame_count = 10

    # Initialize the object detection model
    options = ObjectDetectorOptions(
        num_threads=num_threads,
        score_threshold=0.3,
        max_results=3,
        enable_edgetpu=enable_edgetpu)
    detector = ObjectDetector(model_path=model, options=options)

    # Continuously capture images from the camera and run inference
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit(
                'ERROR: Unable to read from webcam. Please verify your webcam settings.'
            )

        counter += 1
        image = cv2.flip(image, 1)

        # Run object detection estimation using the model.
        detections = detector.detect(image)  # list[detection]
        # detection:bounding_box(left,top,right,bottom),categories(list[Category()label,score,index])
        s_p_detect = 0  # checkpoint
        for d in detections:
            if d.categories[0].label == "stop sign":  # stop sign
                s_p_detect = 1
            elif d.categories[0].label == "person":  # person
                s_p_detect = 2
        cv_detected = s_p_detect
        print(s_p_detect)
        # Draw keypoints and edges on input image
        image = utils.visualize(image, detections)

        # Calculate the FPS
        if counter % fps_avg_frame_count == 0:
            end_time = time.time()
            fps = fps_avg_frame_count / (end_time - start_time)

        # print fps when detecting
            print('Fps now is: ', fps)
            start_time = time.time()

        # Show the FPS
        fps_text = 'FPS = {:.1f}'.format(fps)
        text_location = (left_margin, row_size)
        cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    font_size, text_color, font_thickness)

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('object_detector', image)

    cap.release()
    cv2.destroyAllWindows()
    return


def set_target(rel_y=150, rel_x=0):  # relative position(cm) to car
    global global_map, target_y, target_x
    y = int(rel_y/unit)+curr_y
    x = int(rel_x/unit)+curr_x
    target_x, target_y = bound(x, y)
    global_map[target_y][target_x] = 5
    print('target', target_y, target_x)
    return


def route(dest, start, steps=10):
    path = astar_single(dest, start, steps)
    for operation in path:
        if operation == -1:
            continue
        movement = (operation-curr_dir) % 4
        if movement == 0:
            move_forward()
        elif movement == 1:
            move_left()
        elif movement == 2:
            move_backward()
        elif movement == 3:
            move_right()


class Node(object):
    def __init__(self, prev, loc, direction):
        self.prev = prev
        self.loc = loc
        self.direction = direction

    def __lt__(self, other):
        return self.loc < other.loc


def manhattan_distance(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def neighbors(i, j):
    n = len(global_map)
    m = len(global_map[0])
    res = []
    if i + 1 < n and global_map[i+1][j] not in [1, 2]:
        res.append((i+1, j))
    if i - 1 >= 0 and global_map[i-1][j] not in [1, 2]:
        res.append((i-1, j))
    if j + 1 < m and global_map[i][j+1] not in [1, 2]:
        res.append((i, j+1))
    if j - 1 >= 0 and global_map[i][j-1] not in [1, 2]:
        res.append((i, j-1))

    return res


def astar_single(dest, start, steps=10, limit=10000):
    node = Node(None, start, -1)
    frontier = [(manhattan_distance(start, dest), node)]
    closed = {}
    res = []
    step = 0

    while frontier:
        step += 1
        elem = heapq.heappop(frontier)

        fx = elem[0]
        node = elem[1]  # NODE

        curr = node.loc
        closed[curr] = fx

        from_start = fx - manhattan_distance(curr, dest)

        i = curr[0]
        j = curr[1]
        if step >= limit or (i, j) == dest:
            while node:
                res.append(node.direction)
                node = node.prev
            break

        for neighbor in neighbors(i, j):
            ii, jj = neighbor

            direction = 0  # move down by default

            if jj == j + 1:
                direction = 1  # move right
            if ii == i - 1:
                direction = 2  # move up
            if jj == j - 1:
                direction = 3  # move left
            new_fx = manhattan_distance(
                neighbor, dest) + from_start + 1 + 4*(direction != node.direction)

            if neighbor not in closed or new_fx < closed[neighbor]:
                heapq.heappush(
                    frontier, (new_fx, Node(node, neighbor, direction)))
    res.reverse()
    mlen = min(len(res), steps)
    res = res[:mlen]
    return res


def self_driving():  # self driving until reach target
    set_target()
    count = 0
    while cv_detected == 0 and ((curr_x != target_x) or (curr_y != target_y) and count < 20):
        update_map()
        route((target_y, target_x), (curr_y, curr_x))
        # if cv2.waitKey(1) == 27:  # press esc
        #     break
    print(count)
    plot()
    return


def main():
    # cv_thread = threading.Thread(target=detect, name='cvThread', daemon=True)
    # cv_thread.start()

    for i in range(1):
        self_driving()
        # update_map()
        # # plot()
        # print(cv_thread.name+' is alive ', cv_thread.isAlive())
    fc.stop()
    return


# def main():
#     step_angle = 18
#     global cart_map, global_map, curr_dir
#     np.set_printoptions(threshold=10000, linewidth=1000)

#     polar_map = [[i, 65-0.1*i]
#                  for i in range(-3*step_angle, 3*step_angle+1, step_angle)]
#     update_map(polar_map)
#     move_left()
#     for i in range(20):
#         move_forward()
#     polar_map = [[i, 50-0.1*i]
#                  for i in range(-1*step_angle, 5*step_angle+1, step_angle)]
#     update_map(polar_map)
#     move_right()
#     for i in range(20):
#         move_forward()
#     polar_map = [[i, 35-0.1*i]
#                  for i in range(-2*step_angle, 5*step_angle+1, step_angle)]
#     update_map(polar_map)
#     move_right()
#     for i in range(20):
#         move_forward()
#     polar_map = [[i, 30-0.1*i]
#                  for i in range(-5*step_angle, 5*step_angle+1, step_angle)]
#     update_map(polar_map)
#     for i in range(10):
#         move_backward()
#     polar_map = [[i, 30]
#                  for i in range(-5*step_angle, 5*step_angle+1, step_angle)]
#     move_left()
#     for i in range(20):
#         move_forward()
#     update_map(polar_map)
#     plot()
#     return


if __name__ == "__main__":

    main()
