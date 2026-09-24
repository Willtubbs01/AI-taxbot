class MutationRegistry:
    def __init__(self):
        self._handlers = {}

    def register(self, name, handler):
        if name in self._handlers:
            raise ValueError(f"Duplicate mutation: {name}")
        self._handlers[name] = handler

    def get(self, name):
        return self._handlers[name]
