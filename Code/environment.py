import random
DOORS = [(5, 3), (5, 8), (3, 5), (8, 5)]
ACTIONS =  [(0, 1), (1, 0), (0, -1), (-1, 0)]
ACTION_NAMES = ("up", "right", "down", "left")
FOUR_ROOMS_DIM = 11
class custom_four_rooms():
  def __init__(self, isWindy=False, start_state=(1,1), goal_state=(9,8)):
    self.doors = DOORS
    self.actions = ACTIONS
    self.action_names = ACTION_NAMES
    self.dimensions = FOUR_ROOMS_DIM
    self.isWindy = isWindy
    self.states = self._build_states()
    self.n_states = len(self.states)
    self.n_actions = len(self.actions)
    self.state_to_index = {state: index for index, state in enumerate(self.states)}
    self.start_state = start_state
    self.current_state = self.start_state
    self.goal_state = goal_state

  def _build_states(self):
    def is_internal_wall(pos):
      (x, y) = pos
      if pos in self.doors:
        return False
      if x % 5 == 0 or y % 5 == 0:
        return True
      return False

    states = []
    for i in range(1, self.dimensions-1):
      for j in range(1, self.dimensions-1):
        pos = (i, j)
        if is_internal_wall(pos):
          continue
        states.append(pos)
    return states

  #observation, reward, terminated, truncated, info = env.step(action)
  def step(self, action, state=None):
    if state is None:
      state = self.current_state
    next_state = self.move(state, action)
    reward = 1 if next_state == self.goal_state else 0
    terminated = False if next_state != self.goal_state else True
    truncated = False #wont bother with max iterations just yet?
    info = {} #this is in the gym API.... not req?
    self.current_state = next_state
    return next_state, reward, terminated, truncated, info

  def _move_deterministic(self, state, action):
    x, y = state
    dx, dy = self.actions[action]
    #This line above assumes that action is valid - failing loudly is good because we have messed up if this is not the case!
    next_state = (x + dx, y + dy)
    return next_state if next_state in self.state_to_index else state

  def move(self, state, action):
    if self.isWindy:
      action = self._get_windy_action(action)
    return self._move_deterministic(state, action)

  def _get_windy_action(self, action):
     #From the paper:
     #The chosen action is realised w.p. 1/2, perturbed 90◦ counter clockwise or clockwise each w.p 1/4
     #Actions as defined are moving clockwise as index increases so add one to move clockwise, subtract one otherwise
     actions = [action, action, (action + 1)  % self.n_actions, (action - 1) % self.n_actions]
     return random.choice(actions)



#### VERIFICATION ELEMENTS ####
  def print_room(self, state=(-1,-1)):
    #print a 2D grid of states
    for y in reversed(range(FOUR_ROOMS_DIM)):
      for x in range(FOUR_ROOMS_DIM):
        if (x,y) in self.states:
          if (x,y) == state:
            print('X', end=' ')
          elif (x,y) == self.goal_state:
            print('G', end=' ')
          else:
            print('s', end=' ')
        else:
          if (x,y) in self.doors:
            print('d', end=' ') #This shouldnt happen - only here to catch if doors are passed NOT as states
          else:
            print('.', end=' ')
      print()

  def get_next_valid_states(self, state):
    next_states = set()
    for action in range(self.n_actions):
      next_state = self._move_deterministic(state, action)
      if next_state in self.states and next_state != state:
        next_states.add(next_state)
    return list(next_states)

  def get_transition_probabilities(self, state, action):
    if self.isWindy:
      realised_actions = [
          (action, 0.50),
           ((action + 1) % self.n_actions, 0.25),
            ((action - 1) % self.n_actions, 0.25),
          ]
    else:
      realised_actions = [(action, 1.0)]

    probabilities = {}

    for realised_action, probability in realised_actions:
        next_state = self._move_deterministic(state, realised_action)

        # Multiple realised actions can hit the same wall/state.
        probabilities[next_state] = (
            probabilities.get(next_state, 0.0)
            + probability
        )

    return probabilities

  #This is only for local transitions
  #IF TELEPORTING IS ADDED, THIS NEEDS TO BE UPDATED
  def make_local_supports(self):
    supports = []

    for state in self.states:
        possible_states = {state}

        for action in range(self.n_actions):
            next_state = self._move_deterministic(
                state,
                action,
            )
            possible_states.add(next_state)

        support_indices = [
            self.state_to_index[next_state]
            for next_state in possible_states
        ]

        supports.append(support_indices)

    return supports
