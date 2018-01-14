import random

random.seed(0.1)

EMPTY = "."
MARK_X = "x"
MARK_O = "o"


def seq2grid(seq, size):
    return "\n".join(seq[i * size:(i + 1) * size] for i in range(size))


def toggle_mark(mark):
    if mark == MARK_O:
        return MARK_X
    if mark == MARK_X:
        return MARK_O
    raise ValueError("mark: '" + mark + "' is not among known ones: [" + MARK_O + ", " + MARK_X + "].")


def get_free_positions(state):
    return [i for i, x in enumerate(state) if x == EMPTY]


def make_move(state, pos, mark):
    chars = list(state)
    chars[pos] = mark
    return "".join(chars)


def reward(state, index, size):
    chars = list(state)
    pivot = chars[index]
    index_row = int(index / size)
    index_col = index % size
    row = [index_row * size + i for i in range(size)]
    status = all(elem == pivot for elem in [chars[i] for i in row])
    if status:
        return 1
    col = [index_col + i * size for i in range(size)]
    status = all(elem == pivot for elem in [chars[i] for i in col])
    if status:
        return 1
    if index_row == index_col:
        diag_main = [i * size + i for i in range(size)]
        status = all(elem == pivot for elem in [chars[i] for i in diag_main])
        if status:
            return 1
    if index_row + index_col == size - 1:
        diag_aux = [(size - i - 1) * size + i for i in range(size)]
        status = all(elem == pivot for elem in [chars[i] for i in diag_aux])
        if status:
            return 1
    return 0


N = 3
initState = EMPTY * N * N
states = {}
states[initState] = 0.1

# A single match when the players make random moves
r = 0

mark = MARK_X
state2 = initState
while r == 0:
    free_positions = get_free_positions(state2)
    if len(free_positions) == 0:
        print("tie!")
        break
    pos = random.choice(free_positions)
    state2 = make_move(state2, pos, mark)
    print(seq2grid(state2, N))
    print("")
    r = reward(state2, pos, N)
    if r == 1:
        print("Player " + mark + " won!")
        break
    mark = toggle_mark(mark)

# Single player tic tac toe game
Q = {}
state2 = initState
mark = MARK_X
EPOCHS = 2000
gamma = 0.99
free_positions = get_free_positions(state2)
# initialize available keys in Q table
for free_position in free_positions:
    Q[(state2, free_position, mark)] = random.random()

for e in range(EPOCHS):
    # print("Epoch: " + str(e))
    keys = Q.keys()
    for q in list(keys):
        # print("state: " + q[0] + ", position: " + str(q[1]))
        mark = q[2]
        state1 = q[0]
        state2 = make_move(state1, q[1], mark)
        r = reward(state2, q[1], N)
        # print("new state: " + state + ", reward: " + str(r))
        if r == 1:
            # end of game, there is no next move
            Q[(state2, q[1], mark)] = r
            break
        else:
            free_positions = get_free_positions(state2)
            tmp = []
            for free_position in free_positions:
                key = (state2, free_position, mark)
                if key not in Q:
                    new_state = make_move(state2, free_position, mark)
                    Q[key] = random.random() if reward(new_state, free_position, N) == 0 else 1
                    # print("key " + str(key) + " is added")
                    # print(Q[key])
                tmp.append(Q[key])
            if len(tmp) == 0:
                raise ValueError("list of further Q-entry labels is empty")
            # if (state1, q[1], mark) in Q:
            #    print("old value Q[" + state1 + ", " + str(q[1]) + ", " + mark + "] = " + str(Q[(state1, q[1], mark)]))
            Q[(state1, q[1], mark)] = r + gamma * max(tmp)
            # print("new value: Q[" + state1 + ", " + str(q[1]) + ", " + mark + "] = " + str(Q[(state1, q[1], mark)]))
print(len(Q))

state2 = "........."
free_positions = get_free_positions(state2)
print(max([(Q[(state2, i, MARK_X)], i) for i in free_positions]))
for i in free_positions:
    new_state = make_move(state2, i, MARK_X)
    print(i, Q[(state2, i, MARK_X)])
    print(seq2grid(new_state, N))
