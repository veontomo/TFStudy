import random

random.seed(0.1)

EMPTY = "."
MARK_X = "x"
MARK_O = "o"

N = 3
EMPTY_STATE = EMPTY * N * N


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


def add_to_dict(dict, state, mark, pos, value):
    if state not in dict:
        dict[state] = {}
    if mark not in dict[state]:
        dict[state][mark] = {}
    dict[state][mark][pos] = value


def make_move(state, pos, mark):
    chars = list(state)
    chars[pos] = mark
    return "".join(chars)


def move_reward(state, pos, mark, size):
    new_state = make_move(state, pos, mark)
    r = reward(new_state, pos, size)
    return new_state, r


def reward(state, index, size):
    """ the reward one gets after making a move in position "index" so that the state after the move is "state" """
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


initState = EMPTY_STATE
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
EPOCHS = 20
gamma = 0.6
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

state2 = "xx......."
free_positions = get_free_positions(state2)
print(max([(Q[(state2, i, MARK_X)], i) for i in free_positions]))
for i in free_positions:
    new_state = make_move(state2, i, MARK_X)
    print(i, Q[(state2, i, MARK_X)])
    print(seq2grid(new_state, N))

# Double player tic tac toe game
# Q-table format:
# Q = {".........": {"x": {0: 0.1223, 1: 0.887, ...}}, "x........": {"x": {1: 0.889, 2: 0.25, ...}, "o": {1: 0.223, ...}}}
# so that
# Q[S][M][P] gives the reward for placing the mark M in cell P when the state is S.

Q_X = {}
positions = get_free_positions(EMPTY_STATE)
if EMPTY_STATE not in Q_X:
    Q_X[EMPTY_STATE] = {i: random.random() for i in positions}
print(Q_X)
mark = MARK_X
opponent_mark = toggle_mark(mark)

EPOCHS = 60
for e in range(0, EPOCHS):
    print("epoch " + str(e))
    for state in list(Q_X.keys()):
        for pos in list(Q_X[state]):
            s, r = move_reward(state, pos, mark, N)
            if r == 1:
                # print("move to position " + str(pos) + " in state " + state + " is a winning one")
                # print(Q_X[state])
                Q_X[state][pos] = 1
                break
            current_max = 0
            opponent_positions = get_free_positions(s)
            for opponent_position in opponent_positions:
                state_after_opponent_move, opponent_reward = move_reward(s, opponent_position, opponent_mark, N)
                if state_after_opponent_move not in Q_X:
                    Q_X[state_after_opponent_move] = {i: random.random() for i in
                                                      get_free_positions(state_after_opponent_move)}
                if opponent_reward == 1 and Q_X[state][pos] != -1:
                    Q_X[state][pos] = -1
                tmp = max(Q_X[state_after_opponent_move].values())
                if tmp > current_max:
                    current_max = tmp
            Q_X[state][pos] = r + gamma * current_max
    # print(Q_X[EMPTY_STATE])

print(len(Q_X))
print(Q_X["...xx.oo."])
print(Q_X["........."])
exit()
for state in Q:
    print("state: " + state)
    for mark in Q[state]:
        print("mark: " + str(mark))
        for pos in Q[state][mark]:
            print("pos: " + str(pos) + ", reward: " + str(Q[state][mark][pos]))
            # + ", position: " + str(pos))
            # + ", reward: " + str(Q[state][mark][pos]))
