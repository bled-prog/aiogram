import inspect
from typing import Optional

from ..dispatcher import Dispatcher


class State:
    """
    State object
    """

    def __init__(self, state: Optional[str] = None, group_name: Optional[str] = None):
        self._state = state
        self._group_name = group_name
        self._group = None

    @property
    def state(self):
        if self._group_name is None and self._group:
            group = self._group.__full_group_name__
        elif self._group_name:
            group = self._group_name
        else:
            group = '*'
        return f"{group}:{self._state}"

    def set_parent(self, group):
        if not issubclass(group, StatesGroup):
            raise ValueError('Group must be subclass of StatesGroup')
        self._group = group

    def __set_name__(self, owner, name):
        if self._state is None:
            self._state = name
        self.set_parent(owner)

    def __str__(self):
        return f"<State '{self.state}>'"

    __repr__ = __str__

    async def set(self):
        state = Dispatcher.current().current_state()
        await state.set_state(self.state)


class MetaStatesGroup(type):
    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super(MetaStatesGroup, mcs).__new__(mcs, name, bases, namespace)

        states = []
        childs = []

        cls._group_name = name

        for name, prop in namespace.items():

            if isinstance(prop, State):
                states.append(prop)
            elif inspect.isclass(prop) and issubclass(prop, StatesGroup):
                childs.append(prop)
                prop._parent = cls
            # continue

        cls._parent = None
        cls._childs = tuple(childs)
        cls._states = tuple(states)
        cls._state_names = tuple(state.state for state in states)

        return cls

    @property
    def __group_name__(cls):
        return cls._group_name

    @property
    def __full_group_name__(cls):
        if cls._parent:
            return cls._parent.__full_group_name__ + '.' + cls._group_name
        return cls._group_name

    @property
    def states(cls) -> tuple:
        return cls._states

    @property
    def childs(cls):
        return cls._childs

    @property
    def all_states(cls):
        result = cls.states
        for group in cls.childs:
            result += group.all_states
        return result

    @property
    def all_state_names(cls):
        return tuple(state.state for state in cls.all_states)

    def __str__(self):
        return f"<StatesGroup '{self.__full_group_name__}'>"

    @property
    def state_names(cls) -> tuple:
        return cls._state_names


class StatesGroup(metaclass=MetaStatesGroup):
    @classmethod
    async def next(cls) -> str:
        state = Dispatcher.current().current_state()
        state_name = await state.get_state()

        try:
            next_step = cls.state_names.index(state_name) + 1
        except ValueError:
            next_step = 0

        try:
            next_state_name = cls.states[next_step].state
        except IndexError:
            next_state_name = None

        await state.set_state(next_state_name)
        return next_state_name

    @classmethod
    async def previous(cls) -> str:
        state = Dispatcher.current().current_state()
        state_name = await state.get_state()

        try:
            previous_step = cls.state_names.index(state_name) - 1
        except ValueError:
            previous_step = 0

        if previous_step < 0:
            previous_state_name = None
        else:
            previous_state_name = cls.states[previous_step].state

        await state.set_state(previous_state_name)
        return previous_state_name

    @classmethod
    async def first(cls) -> str:
        state = Dispatcher.current().current_state()
        first_step_name = cls.states[0].state

        await state.set_state(first_step_name)
        return first_step_name

    @classmethod
    async def last(cls) -> str:
        state = Dispatcher.current().current_state()
        last_step_name = cls.states[-1].state

        await state.set_state(last_step_name)
        return last_step_name
