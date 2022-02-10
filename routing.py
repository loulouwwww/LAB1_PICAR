import heapq


class Node(object):
    def __init__(self, prev, loc, direction):
        self.prev = prev
        self.loc = loc
        self.direction = direction

    def __lt__(self, other):
        return self.loc < other.loc


def manhattan_distance(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])


def neighbors(maze, i, j):
    return tuple(x for x in (
        (i + 1, j),
        (i - 1, j),
        (i, j + 1),
        (i, j - 1))
        if maze[x[0]][x[1]] != 1)


def astar_single_one_step(maze, dest, start):
    node = Node(None, start, -1)
    frontier = [(manhattan_distance(start, dest), node)]
    closed = {}

    while frontier:
        elem = heapq.heappop(frontier)

        fx = elem[0]
        node = elem[1]

        curr = node.loc
        closed[curr] = fx

        return_direction = node.direction

        from_start = fx - manhattan_distance(curr, dest)

        x = curr[0]
        y = curr[1]

        neighbors = neighbors(maze, x, y)
        for neighbor in neighbors:
            new_fx = manhattan_distance(neighbor, dest) + from_start + 1

            if neighbor not in closed or new_fx < closed[neighbor]:
                x_0, y_0 = neighbor

                direction = 0  # move down by default

                if x_0 == x + 1:
                    direction = 1  # move right
                if y_0 == y - 1:
                    direction = 2  # move up
                if x_0 == x - 1:
                    direction = 3  # move left

                heapq.heappush(frontier, (new_fx, Node(node, neighbor, direction)))

    return return_direction
