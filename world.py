"""A tiny, safe environment the brain can ACT in and learn from by EXPERIENCE -- the embodiment sandbox.

Text-perceivable state, discrete actions, reward for approaching + reaching a goal. Deterministic (no RNG, so
runs are reproducible and resume-safe). This is the world that closes perceive -> act -> observe -> reward ->
learn with real stakes, instead of learning only by reading. The brain's `experience()` loop drives it.
"""


class GridWorld:
    """A small grid the agent moves around in to reach a goal. Perception is a sentence; actions are compass
    moves; reward is +1 for reaching the goal, +0.1 for getting closer, -0.1 for not. Minimal on purpose --
    the point is the experiential loop, not the task."""

    def __init__(self, size=5, start=(0, 0), goal=None):
        self.size = size
        self.start = list(start)
        self.pos = list(start)
        self.goal = list(goal) if goal else [size - 1, size - 1]
        self.steps = 0

    def reset(self):
        self.pos = list(self.start); self.steps = 0
        return self.perceive()

    def perceive(self):
        """What the agent senses right now, as text it can attend to."""
        dx, dy = self.goal[0] - self.pos[0], self.goal[1] - self.pos[1]
        ew = f"{abs(dx)} {'east' if dx > 0 else 'west'}" if dx else "aligned east-west"
        ns = f"{abs(dy)} {'south' if dy > 0 else 'north'}" if dy else "aligned north-south"
        return (f"I am at ({self.pos[0]},{self.pos[1]}) in a {self.size}x{self.size} grid. "
                f"The goal is {ew} and {ns} of me.")

    def actions(self):
        return ["east", "west", "south", "north"]

    def _dist(self):
        return abs(self.goal[0] - self.pos[0]) + abs(self.goal[1] - self.pos[1])

    def step(self, action):
        """Apply an action, return (reward, done). Reward shaped by whether it got closer -- real consequence."""
        before = self._dist()
        if action == "east":  self.pos[0] = min(self.size - 1, self.pos[0] + 1)
        elif action == "west":  self.pos[0] = max(0, self.pos[0] - 1)
        elif action == "south": self.pos[1] = min(self.size - 1, self.pos[1] + 1)
        elif action == "north": self.pos[1] = max(0, self.pos[1] - 1)
        self.steps += 1
        after = self._dist()
        done = self.pos == self.goal
        reward = 1.0 if done else (0.1 if after < before else -0.1)
        return reward, done

    def optimal_action(self):
        """The move that reduces distance most -- used only to SCORE how well the agent did, never fed to it."""
        dx, dy = self.goal[0] - self.pos[0], self.goal[1] - self.pos[1]
        if abs(dx) >= abs(dy) and dx != 0: return "east" if dx > 0 else "west"
        if dy != 0: return "south" if dy > 0 else "north"
        return "east"
