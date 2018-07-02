#https://codereview.stackexchange.com/questions/85842/a-dictionary-that-allows-multiple-keys-for-one-value

__author__ = 'root-11'

class Database(object):
    """ A dictionary that allows multiple keys for one value """

    def __init__(self):
        self.keys = {}
        self.values = {}

    def __getitem__(self, item):  # <---SQL SELECT statement
        values = self.keys[item]
        if len(values) > 1:
            return sorted(list(values))
        elif len(values) == 1:
            return list(values)[0]

    def __setitem__(self, key, value):
        if key not in self.keys:  # it's a new key <---SQL INSERT statement
            if value not in self.values:  # it's a new value
                self.keys[key] = set()  # a new set
                self.keys[key].add(value)
                self.values[value] = set()  # a new set
                self.values[value].add(key)
            elif value in self.values:
                self.keys[key] = set()  # a new set
                self.keys[key].add(value) # a new key
                self.values[value].add(key)  # but just an update to the values
        elif key in self.keys:  # it's a new relationships
            self.keys[key].add(value)
            if value not in self.values:
                self.values[value] = set()
                self.values[value].add(key)
            elif value in self.values:
                self.values[value].add(key)

    def update(self, key, old_value, new_value):
        """update is a special case because __setitem__ can't see that
        you want to propagate your update onto multiple values. """
        if old_value in self.keys[key]:
            affected_keys = self.values[old_value]
            for key in affected_keys:
                self.__setitem__(key, new_value)
                self.keys[key].remove(old_value)
            del self.values[old_value]
        else:
            raise KeyError("key: {} does not have value: {}".format(key,old_value))

    def __delitem__(self, key, value=None):  # <---SQL DELETE statement
        if value is None:
            # All the keys relations are to be deleted.
            try:
                value_set = self.keys[key]
                for value in value_set:
                    self.values[value].remove(key)
                    if not self.values[value]:
                        del self.values[value]
                del self.keys[key]  # then we delete the key.
            except KeyError:
                raise KeyError("key not found")
        else:  # then only a single relationships is being removed.
            try:
                if value in self.keys[key]:  # this is a set.
                    self.keys[key].remove(value)
                    self.values[value].remove(key)
                if not self.keys[key]:  # if the set is empty, we remove the key
                    del self.keys[key]
                if not self.values[value]:  # if the set is empty, we remove the value
                    del self.values[value]
            except KeyError:
                raise KeyError("key not found")

    def iterload(self, key_list, value_list):
        for key in key_list:
            for value in value_list:
                self.__setitem__(key, value)

def test01():
    # hashable key - a tuple
    d = Database()
    k, v = (1, 2, 3), 'magic'
    d[k] = v
    print(d)
    print(d[k])


def test02():
    # non-hashable key - a list - becomes a many-key shared value
    d = Database()
    keys, value = [1, 2, 3], ['magic']
    d.iterload(keys, value)
    assert d[1] == d[2] == d[3]


def test03():
    d = Database()
    k, v = ['what', 'ever'], ['test']
    d.iterload(k,v)
    assert d['what'] == d['ever']
    d['what'] = 'testing'
    try:
        assert d['what'] == d['ever']
    except AssertionError:
        assert True


def test04():
    d = Database()
    v = 'test'
    keys, values = ['what', 'ever'], [v]
    d.iterload(keys, values)
    d['whatever'] = 'test'
    assert v == d['whatever']
    assert v == d['what']
    assert d['whatever'] == d['what']
    d.update('whatever', 'test', 'new test')
    a, b = d['whatever'], d['what']
    assert a == b


def do_all():
    for k, v in sorted(globals().items()):
        if k.startswith("test") and callable(v):
            v()

if __name__ == "__main__":
    do_all()