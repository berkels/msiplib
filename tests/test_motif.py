import numpy as np
import unittest
from tempfile import TemporaryDirectory
from skimage import img_as_float64
from msiplib.io import read_image
from msiplib.emic import generate_crystal_image
from msiplib.motif import get_motif
from test_unit_cell_from_real_space import count_matches


def reconstruction_matches(image, rec):
    if image.shape != rec.shape:
        raise ValueError("shapes are not matching!")
    else:
        diff = (image - rec) ** 2
        norm_diff = np.sum(diff) / image.size
        return norm_diff


class TestMotifMethods(unittest.TestCase):
    def test_motif(self):
        out_dir = TemporaryDirectory()
        out_dir_name = out_dir.name + "/"

        v = np.array([[20.5, 0], [0, 28.8]])
        im = generate_crystal_image(v)
        params_true_v1 = np.array([0, 20.5])
        params_true_v2 = np.array([28.8, 0])
        print("\n\nim")
        u, reconstruction, v = get_motif(
            im, "im", path=out_dir_name, read_vectors=True, v=np.array([params_true_v1, params_true_v2]), plot=False
        )
        self.assertTrue(reconstruction_matches(im, reconstruction) < 2e-3)
        param_true_list = [params_true_v1, params_true_v2]
        self.assertTrue(count_matches(param_true_list, v, im.shape, 1e-3) == 2)

        v_2 = np.array([[35.6, 0], [0, 18.4]])
        im2 = generate_crystal_image(v_2)
        params_true_v1 = np.array([-18.4, 0])
        params_true_v2 = np.array([0, -35.6])
        print("\n\nim2")
        u2, reconstruction2, v2 = get_motif(
            im2, "im2", path=out_dir_name, read_vectors=True, v=np.array([params_true_v1, params_true_v2]), plot=False
        )
        self.assertTrue(reconstruction_matches(im2, reconstruction2) < 2e-3)
        param_true_list = [params_true_v1, params_true_v2]
        self.assertTrue(count_matches(param_true_list, v2, im2.shape, 1e-3) == 2)

        im3 = read_image("../examples/images/Nc3Nm.nc")
        params_true_v1 = np.array([1.5, 25])
        params_true_v2 = np.array([50, 0])
        print("\n\nim3")
        u3, reconstruction3, v3 = get_motif(
            im3, "im3", path=out_dir_name, read_vectors=True, v=np.array([params_true_v1, params_true_v2]), plot=False
        )
        self.assertTrue(reconstruction_matches(im3, reconstruction3) < 2e-3)
        param_true_list = [params_true_v1, params_true_v2]
        self.assertTrue(count_matches(param_true_list, v3, im3.shape, 1e-3) == 2)

        im4 = read_image("../examples/images/bumps3.nc")
        params_true_v1 = np.array([0.0, 10.0])
        params_true_v2 = np.array([30.0, 0.0])
        print("\n\nim4")
        u4, reconstruction4, v4 = get_motif(
            im4, "im4", path=out_dir_name, read_vectors=True, v=np.array([params_true_v1, params_true_v2]), plot=False
        )
        self.assertTrue(reconstruction_matches(im4, reconstruction4) < 2e-3)
        param_true_list = [params_true_v1, params_true_v2]
        self.assertTrue(count_matches(param_true_list, v4, im4.shape, 1e-3) == 2)

        im5 = read_image("../examples/images/doubleSingle.nc")
        params_true_v1 = np.array([0.0, 15.0])
        params_true_v2 = np.array([25.0, 0.0])
        print("\n\nim5")
        u5, reconstruction5, v5 = get_motif(
            im5, "im5", path=out_dir_name, read_vectors=True, v=np.array([params_true_v1, params_true_v2]), plot=False
        )
        self.assertTrue(reconstruction_matches(im5, reconstruction5) < 2e-3)
        param_true_list = [params_true_v1, params_true_v2]
        self.assertTrue(count_matches(param_true_list, v5, im5.shape, 1e-3) == 2)

        im6 = read_image("../examples/images/hexVacancy.nc")[:, :-9]
        params_true_v1 = np.array([12.0, 12.0])
        params_true_v2 = np.array([12.0, -12.0])
        print("\n\nim6")
        u6, reconstruction6, v6 = get_motif(
            im6, "im6", path=out_dir_name, read_vectors=True, v=np.array([params_true_v1, params_true_v2]), plot=False
        )
        self.assertTrue(reconstruction_matches(im6, reconstruction6) < 2e-3)
        param_true_list = [params_true_v1, params_true_v2]
        self.assertTrue(count_matches(param_true_list, v6, im6.shape, 1e-3) == 2)

        im7 = img_as_float64(read_image("../examples/images/left_grain.png", as_gray=True))
        params_true_v1 = np.array([11.4653, 14.2958])
        params_true_v2 = np.array([18.2864, -2.6899])
        print("\n\nim7")
        u7, reconstruction7, v7 = get_motif(
            im7, "im7", path=out_dir_name, read_vectors=True, v=np.array([params_true_v1, params_true_v2]), plot=False
        )
        self.assertTrue(reconstruction_matches(im7, reconstruction7) < 8e-3)
        param_true_list = [params_true_v1, params_true_v2]
        self.assertTrue(count_matches(param_true_list, np.array(v7), im7.shape, 1) == 2)

        out_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
