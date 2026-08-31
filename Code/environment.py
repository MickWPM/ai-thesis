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


class complex_custom_four_rooms():
  def __init__(self, start_state=(1,1), goal_state=(9,8)):
    self.doors = DOORS
    self.actions = ACTIONS
    self.action_names = ACTION_NAMES
    self.dimensions = FOUR_ROOMS_DIM
    self.states = self._build_states()
    self.n_states = len(self.states)
    self.n_actions = len(self.actions)
    self.state_to_index = {state: index for index, state in enumerate(self.states)}
    self.start_state = start_state
    self.current_state = start_state
    self.goal_state = goal_state

    #room dynamics setup:
    #SW: "breezy" - mmostly reliable but slight chance to move sideways
    #NW: "spin" - akin to biased windy
    #SE: "reversed" - Opposite action executed
    #NE: "Sticky room" - equal chance to take action or stay still. Some slip chance 
    #Probabilities are:
    #intended, clockwise, counter-clockwise, remain still, reversed
    self.room_dynamics = {
      # Highly reliable.
      "south-west": (0.90, 0.05, 0.05, 0.00, 0.00),

      # Strong clockwise drift.
      "north-west": (0.45, 0.45, 0.10, 0.00, 0.00),

      # Action reversal
      "south-east": (0.00, 0.00, 0.00, 0.00, 1.00),

      # Sticky/sluggish room (we would want to avoid this ideally)
      "north-east": (0.40, 0.10, 0.10, 0.40, 0.00),

      # doorways are standard.
      "doorway": (1.00, 0.00, 0.00, 0.00, 0.00),
    }


  def _build_states(self):
    states = []

    for x in range(1, self.dimensions - 1):
      for y in range(1, self.dimensions - 1):
        position = (x, y)
        is_internal_wall = ( (x == 5 or y == 5) and position not in self.doors)

        if not is_internal_wall:
          states.append(position)

    return states
  
  def _get_room(self, state):
    x, y = state
    if x == 5 or y == 5:
      return "doorway"
    if x < 5 and y < 5:
      return "south-west"
    if x < 5 and y > 5:
      return "north-west"
    if x > 5 and y < 5:
      return "south-east"
    return "north-east"

  
  def _get_realised_actions(self, state, action):
    intended, clockwise, counter_clockwise, still, reversed = (self.room_dynamics[self._get_room(state)])

    outcomes = [
      (action, intended),
      ((action + 1) % self.n_actions, clockwise),
      ((action - 1) % self.n_actions, counter_clockwise),
      (None, still),
      ((action - 2) % self.n_actions, reversed),
    ]

    return [(realised_action, probability) for realised_action, probability in outcomes if probability > 0.0]

  def _move_deterministic(self, state, action):
    x, y = state
    dx, dy = self.actions[action]

    next_state = (x + dx, y + dy)

    if next_state in self.state_to_index:
      return next_state

    return state
  

  def move(self, state, action):
    outcomes = self._get_realised_actions(state, action)
    realised_action = random.choices([outcome[0] for outcome in outcomes],
                                     weights=[outcome[1] for outcome in outcomes],
                                     k=1)[0]
    if realised_action is None:
      return state

    return self._move_deterministic(state, realised_action)


  def step(self, action, state=None):
    if state is None:
      state = self.current_state

    next_state = self.move(state, action)
    reward = 1 if next_state == self.goal_state else 0
    terminated = next_state == self.goal_state
    truncated = False
    #We may want to know where we are so lets include that now
    info = {"room": self._get_room(state)} 

    self.current_state = next_state

    return (next_state, reward, terminated, truncated, info)


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
    outcomes = self._get_realised_actions(state, action)
    
    probabilities = {}

    for realised_action, probability in outcomes:
      if realised_action is None:
        next_state = state
      else:
        next_state = self._move_deterministic(state, realised_action)

      probabilities[next_state] = (probabilities.get(next_state, 0.0) + probability)

    return probabilities

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
