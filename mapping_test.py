import heapq
import math

import matplotlib.pyplot as plt
import numpy as np

size = 30  # size of local map
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
target_y = 40
target_x = 40
# Direction:0:screen down(0), 1:screen right(left90), 2:screen up(180), 3:screen left(right 90)
curr_dir = 0
curr_status = 0
# 0:routing 1:reach
# map mark: 0:blank 1:real obs 2:fake obs 3:car center 4:car, 5:target
# assume car w: 3 grids,l: 5 grids
cv_detected = 0


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
    global cart_map, real_local_obs, global_map, fake_obs, curr_x, curr_y
    curr_x, curr_y = bound(curr_x, curr_y)
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
    # bound_f = half_lg-1
    bound_a = half_lg+1
    remove_list = []
    for obs in real_global_obs:
        ly, lx = global_to_local(obs[0], obs[1])
        # print(ly, lx)
        # print(obs[0], obs[1])
        if ly >= 1 and ly <= size and lx >= -half_size+1 and lx <= half_size:
            remove_list.append(obs)
            global_map[obs[0], obs[1]] = 0
            # remove 2 ways
            # for i in range(-bound_f, bound_f+1):
            #     for j in range(-bound_f, bound_f + 1):
            #         if global_map[obs[0]+i][obs[1]+j] == 2:
            #             global_map[obs[0]+i][obs[1]+j] = 0
            for i in range(-bound_a, bound_a+1):
                for j in range(-bound_a, bound_a + 1):
                    # no bound check
                    o_x, o_y = bound(obs[1]+j, obs[0]+i)
                    if i**2+j**2 <= r2 and global_map[o_y][o_x] == 2:
                        global_map[o_y][o_x] = 0
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
        for i in range(-bound_a, bound_a+1):
            for j in range(-bound_a, bound_a + 1):
                # no bound check
                o_x, o_y = bound(base_x+j, base_y+i)
                if i**2+j**2 <= r2 and global_map[o_y][o_x] == 0:
                    global_map[o_y][o_x] = 2
    return


def update_map():
    global cart_map, real_local_obs, global_map
    polar_to_cartesian()
    mark_obs()
    mark_car()
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
    return


def turn_left():
    global curr_dir, curr_y, curr_x
    curr_dir = (curr_dir+1) % 4


def move_left():
    turn_left()
    move_forward()
    mark_car()

    return


def turn_right():
    global curr_dir, curr_y, curr_x
    curr_dir = (curr_dir-1) % 4


def move_right():
    turn_right()
    move_forward()
    mark_car()

    return


def plot():
    plt.figure()
    plt.imshow(global_map)
    plt.show()


def set_target(rel_y=50, rel_x=50):  # relative position(cm) to car
    global curr_status, global_map, target_y, target_x
    curr_status = 0
    # checkpoint
    y = int(rel_y/unit)+curr_y
    x = int(rel_x/unit)+curr_x
    target_x, target_y = bound(x, y)
    global_map[y][x] = 5
    return


def test():
    return


def route(dest, start, steps=5):
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


def astar_single(dest, start, steps=10, limit=2000):
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
            print('explore steps', step)
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
                neighbor, dest) + from_start + 1 + 2*(direction != node.direction)

            if neighbor not in closed or new_fx < closed[neighbor]:
                heapq.heappush(
                    frontier, (new_fx, Node(node, neighbor, direction)))
    res.reverse()
    mlen = min(len(res), steps)
    res = res[:mlen]
    return res


def main():
    step_angle = 18
    global polar_map, cart_map, global_map, curr_dir
    np.set_printoptions(threshold=10000, linewidth=1000)

    # polar_map = [[i, 20-0.1*i]
    #              for i in range(-3*step_angle, 3*step_angle+1, step_angle)]

    # update_map()
    # plot()

    # turn_left()
    # polar_map = []
    # update_map()
    # plot()
    # move_right()
    # update_map()
    # plot()

    # print((curr_y, curr_x))
    # update_map()
    # for i in range(4):
    #     curr_dir = i
    #     y1, x1 = local_to_global(5, 2)
    #     print(y1, x1)
    #     y2, x2 = global_to_local(y1, x1)
    #     print(y2, x2)
    #     global_map[y1][x1] = 9
    # plot()

    polar_map = [[i, 70-0.1*i]
                 for i in range(-3*step_angle, 3*step_angle+1, step_angle)]

    set_target(110, 0)
    update_map()
    plot()
    # route((target_y, target_x), (curr_y, curr_x))
    move_right()
    for i in range(4):
        move_forward()

    polar_map = [[i, 20-0.1*i]
                 for i in range(-5*step_angle, 1*step_angle+1, step_angle)]
    update_map()
    plot()
    count = 0
    while (curr_y, curr_x) != (target_y, target_x) and count < 20:
        route((target_y, target_x), (curr_y, curr_x))
        count += 1
    print('time of calling route:', count)
    plot()

    return


if __name__ == "__main__":
    main()
