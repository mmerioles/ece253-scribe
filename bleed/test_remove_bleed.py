"""Small invariants for geometric and destructive-mask errors; no OCR required."""
import unittest
from unittest.mock import patch
import remove_bleed
import cv2
import numpy as np
from remove_bleed import register, restore, page_mask, BG, FRONT, BACK, BOTH


class BleedTests(unittest.TestCase):
    def test_registration_fallback_keeps_native_coordinates(self):
        image = np.full((1800, 1200, 3), 180, np.uint8)
        report = dict(homography=np.eye(3).tolist(), median_error_pixels=0.5)
        with patch.object(remove_bleed, '_register', side_effect=[ValueError('native failed'), (None, None, report)]) as matching:
            aligned, valid, result = register(image, image, already_mirrored=True)
        self.assertEqual(matching.call_count, 2)
        self.assertEqual(aligned.shape, image.shape)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(result['homography'], np.eye(3), atol=1e-12)
        self.assertEqual(result['feature_max_side'], 1600)

    def test_registration_fallback_rejects_weak_matches(self):
        image = np.full((1800, 1200, 3), 180, np.uint8)
        with patch.object(remove_bleed, '_register', side_effect=ValueError('weak')):
            with self.assertRaises(ValueError):
                register(image, image)

    def test_dark_ink_inside_page_remains_valid(self):
        image=np.zeros((120,100,3),np.uint8)
        image[10:110,10:90]=180
        image[45:65,40:60]=15
        valid=page_mask(image)
        self.assertTrue(valid[55,50])
        self.assertFalse(valid[0,0])

    def test_reverse_is_mirrored_exactly_once(self):
        rng = np.random.default_rng(3)
        front = np.full((700,550,3),220,np.uint8)
        for _ in range(250):
            x,y = rng.integers([25,25],[525,675]); c=int(rng.integers(25,190))
            cv2.circle(front,(int(x),int(y)),int(rng.integers(3,9)),(c,c,c),-1)
        H = np.array([[1,.005,8],[.003,1,-7],[.00001,0,1]],float)
        reverse = cv2.warpPerspective(front,H,(550,700),borderValue=(220,220,220))[:,::-1].copy()
        aligned,valid,report = register(front,reverse)
        interior=valid.copy(); interior[:30]=0;interior[-30:]=0;interior[:,:30]=0;interior[:,-30:]=0
        self.assertLess(np.mean(np.abs(front[interior].astype(float)-aligned[interior])),5)
        self.assertGreater(report['inliers'],30)
        second,_,_=register(front,reverse[:,::-1].copy(),True)
        np.testing.assert_array_equal(second,aligned)

    def test_blank_pair_rejected(self):
        blank=np.full((500,500,3),230,np.uint8)
        with self.assertRaises(ValueError): register(blank,blank)

    def test_overlap_and_invalid_pixels_are_not_erased(self):
        image=np.full((80,100,3),220,np.uint8)
        labels=np.full((80,100),BG,np.uint8)
        labels[20:55,10:20]=FRONT; labels[20:55,45:55]=BACK; labels[20:55,75:85]=BOTH
        image[labels!=BG]=60
        valid=np.ones(labels.shape,bool);valid[:5]=False
        darkness=cv2.cvtColor(255-image,cv2.COLOR_RGB2GRAY)
        output,mask,_=restore(image,labels,darkness,darkness,valid)
        np.testing.assert_array_equal(output[labels==FRONT],image[labels==FRONT])
        np.testing.assert_array_equal(output[labels==BOTH],image[labels==BOTH])
        np.testing.assert_array_equal(output[~valid],image[~valid])
        self.assertGreater(output[30:45,47:53].mean(),180)
        self.assertFalse(mask[~valid].any())


if __name__=='__main__': unittest.main()
