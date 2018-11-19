import unittest
import os

from fileio import FileIO
from cluster import Cluster
from point import Point

class TestPoint(unittest.TestCase):

    # comparing 2 same points
    def test__eq__same_points(self):
        coordinates = [1, 2, 3, 4, 5, 6, 7, 8]
        a = Point(coordinates)
        b = Point(coordinates)
        self.assertEqual((a == b), True)

    # comparing 2 different points
    def test__eq__different_points(self):
        coordinates = [1, 2, 3, 4, 5, 6, 7, 8]
        a = Point(coordinates)
        b = Point([1, 1])
        self.assertEqual((a == b), False)

class TestCluster(unittest.TestCase):

    def test__eq__same_clusters(self):
        points = [
            Point([1, 1, 1]),
            Point([12, 1, 1]),
            Point([4, 1, 1]),
            Point([0, 1, 1]),
            Point([3, 1, 1]),
            Point([2, 1, 1])
        ]
        a = Cluster(0, points)
        b = Cluster(0, points)
        self.assertEqual(a == b, True)

    def test__eq__different_clusters(self):
        a = Cluster(0, [Point([1, 1, 1])])
        b = Cluster(1, [Point([1, 1, 1])])
        self.assertEqual(a == b, False)
        c = Cluster(0, [Point([1, 1, 1])])
        d = Cluster(0, [Point([0, 10, 1])])
        self.assertEqual(c == d, False)


class TestFileIO(unittest.TestCase):

    def test___init___file_does_not_exist(self):
        with self.assertRaises(Exception):
            FileIO("foo", "r")

    def test___init___file_exists(self):
        if (os.path.exists("foo")):
            os.remove("foo")

        f = open("foo", "w")
        f.close()
        with self.assertRaises(Exception):
            FileIO("foo", "w")
    
    def test_close_file(self):
        if (os.path.exists("foo")):
            os.remove("foo")

        f = FileIO("foo", "w")
        with self.assertRaises(Exception):
            f.close()

   # def test_loadFile_good_file(self):
   #     f = FileIO("tests/test1.csv", "r")
   #     loaded = f.loadFile()
   #     reference = [
   #         Cluster(0, [Point([1, 1, 1]), Point([2, 2, 2]), Point([3.0, 3.0, 3.0])]),
   #         Cluster(1, [Point([1, 1, 1]), Point([1, 1, 1]), Point([1, 1, 1]),
   #             Point([1, 1, 1]), Point([1, 1, 1]), Point([1, 1, 1]), ]),
   #         Cluster(2, [Point([1, 1, 1])])
   #     ] #TODO compare cluster by cluster
   #     self.assertEquals(loaded, reference)


if __name__ == '__main__':
    unittest.main()