import unittest
import os

from fileio import FileIO
from cluster import Cluster
from point import Point

# testcases for point.py
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

    # further testing __eq__ by trying to remove from list
    def test_removing_from_list(self):
        points = [
            Point([1, 1, 1]),
            Point([12, 1, 1]),
            Point([4, 1, 1]),
            Point([0, 1, 1]),
            Point([3, 1, 1]),
            Point([2, 1, 1])
        ]
        points.remove(points[0])
        points_removed = [
            Point([12, 1, 1]),
            Point([4, 1, 1]),
            Point([0, 1, 1]),
            Point([3, 1, 1]),
            Point([2, 1, 1])
        ]
        self.assertEqual(points == points_removed, True)


# testcases for cluster.py
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

# testcases for fileio.py
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

    def test_loadFile_good_file(self):
        f = FileIO("tests/test1.csv", "r")
        loaded = f.loadFile()
        f.close()

        reference = [
            Cluster('0', [Point([1, 1, 1]), Point([2, 2, 2]), Point([3.0, 3.0, 3.0])]),
            Cluster('1', [Point([1, 1, 1]), Point([1, 1, 1]), Point([1, 1, 1]),
                Point([1, 1, 1]), Point([1, 1, 1]), Point([1, 1, 1]), ]),
            Cluster('2', [Point([1, 1, 1])])
        ]
        self.assertEqual(loaded, reference)

    def test_loadFile_wrong_file(self):
        f1 = FileIO("tests/test1.csv", "r")
        loaded1 = f1.loadFile()
        f1.close()

        f2 = FileIO("tests/test2.csv", "r")
        loaded2 = f2.loadFile()
        f2.close()
        self.assertEqual((loaded1 == loaded2), False)        


if __name__ == '__main__':
    unittest.main()