"""
Custom collections

classes:
    * IndexedDeque
    * ObservableDict
"""

from collections import deque
from collections import defaultdict


class IndexedDeque:
    """
    Deque with set.
    Set is using for check for unique values.

    Parameters
    -------
    size: int
        max deque size

    Attributes
    -------
    deque: deque
        store values
    set: set
        check for unique values

    Examples
    -------
    >>> rl = IndexedDeque(3)
    >>> rl.appendleft('1')
    >>> rl.appendleft('2')
    >>> rl.appendleft('3')
    >>> rl.appendleft('4')
    >>> rl.deque
    deque(['4', '3', '2'], maxlen=3)
    """

    def __init__(self, size: int) -> None:
        self.deque = deque(maxlen=size)
        self.set = set()

    def appendleft(self, value) -> None:
        """
        Append left value to deque.
        Add value to set
        """
        if value in self.set:
            return
        if len(self.deque) == self.deque.maxlen:
            discard = self.deque.pop()
            self.set.discard(discard)
        self.deque.appendleft(value)
        self.set.add(value)


class ObservableDict:
    """
    A dictionary wrapper that triggers callbacks on changes.

    This class wraps around a standard Python dictionary and allows for
    the registration of callback functions that are triggered whenever
    the dictionary is modified (e.g., item addition, deletion, or update).

    Attributes:
        _data (dict): The internal dictionary to store data.
        _callbacks (list): A list of callback functions to call on changes.
    """

    def __init__(self, initial_dict=None):
        """
        Initialize the ObservableDict with an optional initial dictionary.

        Args:
            initial_dict (dict, optional): An initial dictionary to populate
                                            the ObservableDict. Defaults to None.
        """
        self._data = initial_dict or {}
        self._callbacks = []

    def on_change(self, callback):
        """
        Register a callback function to be called when the dictionary changes.

        Args:
            callback (function): A callback function that takes the updated
                                 dictionary as its parameter.
        """
        self._callbacks.append(callback)

    def _trigger_callbacks(self):
        """
        Call all registered callback functions with the current state of the dictionary.
        """
        for callback in self._callbacks:
            callback(self._data)

    def __setitem__(self, key, value):
        """
        Set the value for a key in the dictionary and trigger callbacks.

        Args:
            key: The key for which to set the value.
            value: The value to set for the specified key.
        """
        self._data[key] = value
        self._trigger_callbacks()

    def __getitem__(self, key):
        """
        Get the value associated with a key in the dictionary.

        Args:
            key: The key to look up.

        Returns:
            The value associated with the key.
        """
        return self._data[key]

    def __delitem__(self, key):
        """
        Delete a key-value pair from the dictionary and trigger callbacks.

        Args:
            key: The key to delete.
        """
        del self._data[key]
        self._trigger_callbacks()

    def __contains__(self, key):
        """
        Check if a key is in the dictionary.

        Args:
            key: The key to check for.

        Returns:
            bool: True if the key is in the dictionary, False otherwise.
        """
        return key in self._data

    def __iter__(self):
        """
        Return an iterator over the keys of the dictionary.

        Returns:
            iterator: An iterator over the keys of the dictionary.
        """
        return iter(self._data)

    def __len__(self):
        """
        Return the number of items in the dictionary.

        Returns:
            int: The number of items in the dictionary.
        """
        return len(self._data)

    def update(self, *args, **kwargs):
        """
        Update the dictionary with the key-value pairs from other,
        overwriting existing keys, and trigger callbacks.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self._data.update(*args, **kwargs)
        self._trigger_callbacks()

    def clear(self):
        """
        Remove all items from the dictionary and trigger callbacks.
        """
        self._data.clear()
        self._trigger_callbacks()

    def keys(self):
        """
        Return a new view of the dictionary's keys.

        Returns:
            dict_keys: A view of the dictionary's keys.
        """
        return self._data.keys()

    def values(self):
        """
        Return a new view of the dictionary's values.

        Returns:
            dict_values: A view of the dictionary's values.
        """
        return self._data.values()

    def items(self):
        """
        Return a new view of the dictionary's items (key, value pairs).

        Returns:
            dict_items: A view of the dictionary's items.
        """
        return self._data.items()

    def get(self, key, default=None):
        """
        Return the value for a key if it exists, else a default value.

        Args:
            key: The key to look up.
            default: The value to return if the key is not found. Defaults to None.

        Returns:
            The value associated with the key if it exists, else the default value.
        """
        return self._data.get(key, default)

    def setdefault(self, key, default=None):
        """
        If the key is in the dictionary, return its value.
        If not, insert the key with a value of default and trigger callbacks.

        Args:
            key: The key to look up or add.
            default: The value to set if the key is not found. Defaults to None.

        Returns:
            The value associated with the key if it exists, else the default value.
        """
        result = self._data.setdefault(key, default)
        self._trigger_callbacks()
        return result

    def get_data(self) -> dict:
        """
        Returns _data

        Returns:
        -------
        dict
        """
        return self._data


class NestedDefaultDict:
    """
    A nested defaultdict structure with three levels of nesting, where the innermost
    level defaults to float.

    This class encapsulates a nested defaultdict structure with a convenient API for
    accessing and setting nested dictionary values.

    Examples
    --------
    >>> from src.data.collections import NestedDefaultDict
    >>> a = NestedDefaultDict()
    >>> a['key1']['key2']['key3'] = 1.0
    >>> print(a['key1']['key2']['key3'])
    1.0
    >>> print(a)
    defaultdict(<bound method NestedDefaultDict._nested_defaultdict of <__main__.NestedDefaultDict object at ...>>,
               {'key1': defaultdict(<bound method NestedDefaultDict._float_defaultdict of <__main__.NestedDefaultDict object at ...>>,
                                    {'key2': defaultdict(<class 'float'>, {'key3': 1.0})})})

    Attributes
    ----------
    data : defaultdict
       The main nested defaultdict structure.

    Methods
    -------
    __getitem__(key)
       Returns the value associated with the key, creating nested levels as needed.
    __setitem__(key, value)
       Sets the value associated with the key.
    __contains__(key)
        Checks if the key exists in the nested structure.
    __repr__()
       Returns the string representation of the object.
    __str__()
       Returns the string representation of the object.
    items()
        Returns an iterator over the (key, value) pairs in the nested structure.
    """

    def __init__(self):
        """
        Initializes the NestedDefaultDict with a three-level nested defaultdict structure.
        """
        self.data = defaultdict(self._nested_defaultdict)

    def _float_defaultdict(self):
        """
        Returns a defaultdict that defaults to float.

        Returns
        -------
        defaultdict
            A defaultdict that returns float by default.
        """
        return defaultdict(float)

    def _nested_defaultdict(self):
        """
        Returns a defaultdict that defaults to another level of nested defaultdict.

        Returns
        -------
        defaultdict
            A defaultdict that returns a defaultdict with float as the innermost default.
        """
        return defaultdict(self._float_defaultdict)

    def __getitem__(self, key):
        """
        Returns the value associated with the key, creating nested levels as needed.

        Parameters
        ----------
        key : hashable
            The key to access in the nested defaultdict.

        Returns
        -------
        defaultdict
            The value associated with the key, which is another level of defaultdict.
        """
        return self.data[key]

    def __setitem__(self, key, value):
        """
        Sets the value associated with the key.

        Parameters
        ----------
        key : hashable
            The key to set in the nested defaultdict.
        value : any
            The value to associate with the key.
        """
        self.data[key] = value

    def __delitem__(self, key):
        """
        Delete a key-value pair from the dictionary and trigger callbacks.

        Args:
            key: The key to delete.
        """
        del self.data[key]

    def __contains__(self, key):
        """
        Checks if the key exists in the nested structure without creating new entries.

        Parameters
        ----------
        key : hashable
            The key to check in the nested defaultdict.

        Returns
        -------
        bool
            True if the key exists, False otherwise.
        """
        return key in self.data

    def __repr__(self):
        """
        Returns the string representation of the object.

        Returns
        -------
        str
            The string representation of the NestedDefaultDict object.
        """
        return repr(self.data)

    def __str__(self) -> str:
        """
        Returns the string representation of the object.

        Returns
        -------
        str
            The string representation of the NestedDefaultDict object.
        """
        return str(self._recursive_dict(self.data))

    def _recursive_dict(self, d):
        """
        Recursively converts the nested defaultdict structure to a regular dictionary
        for pretty printing.

        Parameters
        ----------
        d : defaultdict
            The nested defaultdict to convert.

        Returns
        -------
        dict
            The converted dictionary.
        """
        if isinstance(d, defaultdict):
            return {k: self._recursive_dict(v) for k, v in d.items()}
        return d

    def items(self):
        """
        Returns an iterator over the (key, value) pairs in the nested structure.

        Yields
        ------
        tuple
            A tuple containing the key and value, where value may be another defaultdict.
        """
        return self._recursive_items(self.data)

    def _recursive_items(self, d):
        """
        Recursively yields (key, value) pairs from the nested defaultdict structure.

        Parameters
        ----------
        d : defaultdict
            The nested defaultdict to iterate over.

        Yields
        ------
        tuple
            A tuple containing the key and value, where value may be another defaultdict.
        """
        for k, v in d.items():
            if isinstance(v, defaultdict):
                yield k, dict(self._recursive_items(v))
            else:
                yield k, v
