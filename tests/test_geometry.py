"""CPU tests for geometry, mask invariants and invalid input handling."""
import sys
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import logic


class GeometryTests(unittest.TestCase):
    def setUp(self):
        pixels = np.zeros((128, 192, 3), dtype=np.uint8)
        pixels[:, :, 0] = np.arange(192)
        pixels[:, :, 1] = np.arange(128)[:, None]
        self.image = Image.fromarray(pixels)

    def test_expansion_directions_preserve_source_and_core(self):
        for direction, size, offset in [
            ('left', (256,128), (64,0)), ('right',(256,128),(0,0)),
            ('top',(192,192),(0,64)), ('bottom',(192,192),(0,0)),
            ('all',(320,256),(64,64))]:
            with self.subTest(direction=direction):
                canvas, mask = logic.prepare_outpainting(self.image,64,direction,overlap=16)
                self.assertEqual(canvas.size,size)
                x,y=offset
                np.testing.assert_array_equal(np.asarray(canvas.crop((x,y,x+192,y+128))),np.asarray(self.image))
                self.assertEqual(mask.getpixel((x+96,y+64)),0)
                self.assertEqual(set(np.unique(mask)),{0,255})
                if direction in ('right','all'):
                    self.assertEqual(mask.getpixel((x+191,y+64)),255)
                if direction == 'right':
                    self.assertEqual(mask.getpixel((0,64)),0)

    def test_no_overlap_preserves_every_original_mask_pixel(self):
        canvas, mask=logic.prepare_outpainting(self.image,64,'all',overlap=0)
        self.assertFalse(np.any(np.asarray(mask)[64:192,64:256]))
        self.assertTrue(np.all(np.asarray(mask)[:64] == 255))

    def test_invalid_geometry(self):
        for kwargs in [dict(direction='diagonal'),dict(expand_pixels=17),dict(overlap=128)]:
            with self.assertRaises(ValueError): logic.prepare_outpainting(self.image,**kwargs)
        with self.assertRaises(ValueError): logic.normalize_image(Image.new('RGB',(513,512)))

    def test_manual_mask_convention(self):
        mask=logic.manual_mask(self.image,(100,20,170,100))
        self.assertEqual(mask.mode,'L')
        self.assertEqual(mask.getpixel((10,10)),0)
        self.assertEqual(mask.getpixel((120,50)),255)

    def test_parameter_boundaries(self):
        logic._validate('astronaut',222,30,7.5,4)
        for args in [('',1,30,7.5,1),('x',-1,30,7.5,1),('x',1,0,7.5,1),
                     ('x',1,30,21,1),('x',4294967295,30,7.5,4)]:
            with self.assertRaises(ValueError): logic._validate(*args)


if __name__ == '__main__': unittest.main()
