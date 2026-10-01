import unittest
import numpy as np
from boardsesh_import import decode_frames


class FramesTest(unittest.TestCase):
    def test_roles_and_positions(self):
        mapping = {101: (0, 1), 102: (1, 1), 103: (2, 1), 104: (3, 1)}
        a = decode_frames('p101r12p102r14p103r13p104r15', mapping, (4, 2))
        self.assertEqual(a.dtype, np.float32)
        self.assertEqual([a[i, 1, i] for i in range(4)], [1]*4)
        self.assertEqual(a.sum(), 4)

    def test_reject_lossy_or_ambiguous_conversion(self):
        mapping = {1: (0, 0), 2: (1, 0)}
        for text in ['p1r12p2r14p3r13', 'p1r42p2r14', 'p1r12p1r14',
                     'p1r12,p2r14', 'p1r12p2r14junk', 'p1r13p2r14']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                decode_frames(text, mapping, (2, 1))
