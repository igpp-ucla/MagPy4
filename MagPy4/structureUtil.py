class CircularList(list):
    def next_val(self):
        if len(self) == 0:
            return None
        else:
            val = self.pop(0)
            self.append(val)
            return val