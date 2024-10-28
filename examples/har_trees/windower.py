
import array

def empty_array(typecode, length, value=0):
    return array.array(typecode, (value for _ in range(length)))

def shift_array(arr, shift):
    for i in range(shift, len(arr)):
        arr[i-shift] = arr[i]


class TriaxialWindower():
    def __init__(self, length):
        self.length = length
        self.x_values = empty_array('h', length)
        self.y_values = empty_array('h', length)
        self.z_values = empty_array('h', length)

        self._valid = 0

    def full(self):
        return self._valid == self.length

    def push(self, xs, ys, zs):
        hop = len(xs)
        assert len(xs) == hop
        assert len(ys) == hop
        assert len(zs) == hop

        remaining = self.length - self._valid
        if remaining <= hop:
            # make space for new data
            shift = hop
            shift_array(self.x_values, shift)
            shift_array(self.y_values, shift)
            shift_array(self.z_values, shift)
            insert = self.length - shift
        else:
            insert = self._valid

        self._valid = insert + hop
        if self._valid > self.length:
            raise ValueError("Hop must be multiple of length")


        self.x_values[insert:insert+hop] = xs
        self.y_values[insert:insert+hop] = ys
        self.z_values[insert:insert+hop] = zs

