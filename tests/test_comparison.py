import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from comparison_data import make_split, manifest_subsets
from evaluate_comparison import route_scores


class ComparisonTests(unittest.TestCase):
    def test_split_keeps_role_variants_together(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for i in range(30):
                a = np.zeros((6, 6, 4), np.float32)
                a.flat[(i // 2)*4+i%2] = 1
                np.save(root / f'{i}.npy', a)
                (root / f'{i}.json').write_text(json.dumps({'grade_v': 5}))
            result = make_split(root, root/'split.json')
            self.assertEqual(result, make_split(root, root/'split.json'))
            partition = {stem: k for k, stems in result['splits'].items() for stem in stems}
            for i in range(0, 30, 2):
                self.assertEqual(partition[str(i)], partition[str(i+1)])
            ds = SimpleNamespace(samples=[SimpleNamespace(npy_path=root/f'{i}.npy') for i in range(30)])
            subsets = manifest_subsets(ds, root/'split.json')
            self.assertEqual(sum(len(s) for s in subsets), 30)
            result['splits']['test'].append(result['splits']['train'][0])
            (root/'split.json').write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                manifest_subsets(ds, root/'split.json')

    def test_reach_excludes_feet(self):
        a = np.zeros((1, 3, 4), bool)
        a[0, 0, 0] = a[0, 1, 2] = a[0, 2, 1] = True
        xy = np.array([[0., 0.], [0., 8.], [0., 16.]])
        self.assertEqual(route_scores(a, xy, 1)['finish_reachable_rate'], 1.)
        a[0, 1, 2] = False
        a[0, 1, 3] = True
        self.assertEqual(route_scores(a, xy, 1)['finish_reachable_rate'], 0.)


if __name__ == '__main__':
    unittest.main()
