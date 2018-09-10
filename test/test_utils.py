
import unittest
import numpy as np
from multiprocessing import Array

from senseact.utils import get_random_state_array, get_random_state_from_array


class TestUtils(unittest.TestCase):

    def test_random_state_array(self):
        rand_obj = np.random.RandomState(1)
        rand_state = rand_obj.get_state()
        original_uniform_values = rand_obj.uniform(-1, 1, 100)
        original_normal_values = rand_obj.randn(100)

        rand_state_array_type, rand_state_array_size, rand_state_array = get_random_state_array(rand_state)
        shared_rand_array = np.frombuffer(Array('b', rand_state_array_size).get_obj(), dtype=rand_state_array_type)
        np.copyto(shared_rand_array, np.frombuffer(rand_state_array, dtype=rand_state_array_type))

        new_rand_obj = np.random.RandomState()
        new_rand_obj.set_state(get_random_state_from_array(shared_rand_array))
        new_uniform_values = new_rand_obj.uniform(-1, 1, 100)
        new_normal_values = new_rand_obj.randn(100)

        assert np.all(original_uniform_values == new_uniform_values)
        assert np.all(original_normal_values == new_normal_values)


if __name__ == '__main__':
    unittest.main(buffer=True)
