"""Invariants for the overlap decomposition; synthetic, no OCR required."""
import unittest

import cv2
import numpy as np

import recover_overlap as ro


class OverlapTests(unittest.TestCase):
    def setUp(self):
        h, w = 120, 200
        self.m_back = np.zeros((h, w), np.float32)
        cv2.rectangle(self.m_back, (40, 30), (160, 90), 1, -1)          # opaque reverse stroke
        self.gain = np.full((h, w), .6, np.float32)
        self.pencil = np.zeros((h, w), np.float32)
        cv2.line(self.pencil, (10, 60), (190, 60), .4, 5)                 # pencil crossing it
        self.region = cv2.dilate((self.m_back > 0).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        self.params = {**ro.DEFAULTS, 'gain_sigma': 6, 'lam0': .05, 'lam_ink': .05, 'lam_edge': 0,
                       'min_area': 0, 'back_sigma': 0}

    def test_pencil_inside_bleed_is_recovered_and_bleed_removed(self):
        d = self.gain*self.m_back + self.pencil
        p, a, history = ro.decompose(d, self.m_back, self.region, self.params)
        inside = self.region & (self.m_back > 0)
        stroke = inside & (self.pencil > 0)
        self.assertGreater(p[stroke].mean(), .3)                          # pencil kept
        self.assertLess(p[inside & (self.pencil == 0)].max(), .05)        # bleed removed
        self.assertLessEqual(history[-1], history[0] + 1e-6)              # objective does not increase

    def test_l0_keeps_unbiased_density(self):
        d = self.gain*self.m_back + self.pencil
        p, _, _ = ro.decompose(d, self.m_back, self.region, {**self.params, 'penalty': 'l0'})
        stroke = self.region & (self.m_back > 0) & (self.pencil > 0)
        self.assertAlmostEqual(float(np.median(p[stroke])), .4, delta=.05)

    def test_isolated_speckle_dropped_but_entering_stroke_kept(self):
        d = self.pencil.copy()
        p = np.zeros_like(d)
        p[60-2:60+3, 40:160] = .4                                         # continues pencil from outside
        p[40:43, 100:103] = .4                                            # isolated 9 px speckle
        out = ro.stroke_continuity(p, p, d, self.region, {**self.params, 'min_area': 50})
        self.assertTrue((out[60, 40:160] > 0).all())
        self.assertFalse(out[40:43, 100:103].any())

    def test_grid_removed_but_slanted_stroke_kept(self):
        D = np.zeros((150, 150), np.float32)
        D[40, :] = .15; D[:, 70] = .15                                    # printed grid
        cv2.line(D, (20, 130), (60, 20), .4, 3)                           # handwriting
        out = ro.remove_grid(D, 61)
        self.assertLess(out[40, 120], .01); self.assertLess(out[120, 70], .01)
        self.assertGreater(out[75, 40], .3)

    def test_gap_continuation_bridges_opaque_core_only(self):
        p = np.zeros((60, 120), np.float32); p[28:32, :] = .4
        core = np.zeros(p.shape, bool); core[15:45, 50:66] = True
        p[core] = 0                                                       # stroke hidden by opaque bleed
        out = ro.continue_gaps(p, core, 7)
        self.assertGreater(out[30, 58], .1)                               # bridged (faded) inside the core
        np.testing.assert_array_equal(out[~core], p[~core])               # nothing changes outside

    def test_minimal_restore_guarantees(self):
        rng = np.random.default_rng(0)
        h, w = 120, 200
        front = np.clip(200 + rng.normal(0, 3, (h, w, 3)), 0, 255).astype(np.uint8)
        front[58:62, 10:190] = 90                                         # front stroke
        cv2.rectangle(front, (60, 30), (140, 90), (120, 120, 120), -1)    # bleed blob over it
        front[58:62, 60:140] = 60                                         # stroke inside the bleed
        region = np.zeros((h, w), bool); region[25:95, 55:145] = True
        bleed = np.zeros((h, w), np.float32); bleed[30:91, 60:141] = .5
        pencil = np.zeros((h, w), np.float32); pencil[58:62, 60:141] = .4
        state = dict(front=front, valid=np.ones((h, w), bool), d_front=ro.density(front),
                     d_back=bleed*3, plate=np.full_like(front, 200))
        result = dict(region=region, bleed=bleed, pencil=pencil)
        for mode in ('off', 'on', 'divide'):
            out = ro.minimal_restore(state, result, protection=mode, paths=False)
            touched, ink = out['touched'], out['ink']
            np.testing.assert_array_equal(out['output'][~touched], front[~touched])   # bit-identical elsewhere
            if mode == 'on':
                self.assertTrue(ink.any())
                np.testing.assert_array_equal(out['output'][ink], front[ink])         # strokes untouched
            self.assertGreater(out['output'][40, 100].mean(), 170)                    # bleed-only -> paper


if __name__ == '__main__':
    unittest.main()
