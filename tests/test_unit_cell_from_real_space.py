"""
This script is for testing :py:meth:`get_primitive_unit_cell_vectors <msiplib.unit_cell_from_real_space.get_primitive_unit_cell_vectors>`
"""
import numpy as np
import unittest
from tempfile import TemporaryDirectory
from msiplib.unit_cell_from_real_space import get_primitive_unit_cell_vectors
from msiplib.io import read_image
from msiplib.emic import generate_crystal_image


def count_matches(param_true, vectors, grid_shape, tol=1e-3):
    match_list = []
    for i in range(len(vectors)):
        if (np.linalg.norm(vectors[i] - param_true[i]) <= tol) or (np.linalg.norm(vectors[i] + param_true[i]) <= tol):
            match_list.append([vectors[i], param_true[i]])
    return len(match_list)


def print_vectors(v1, v2, params_true_v1, params_true_v2):
    string_formatter = {"float_kind": lambda x: f"{x:.2f}"}
    print(
        "calculated vectors: ",
        np.array2string(v1, formatter=string_formatter),
        np.array2string(v2, formatter=string_formatter),
        "\ninput vectors: ",
        params_true_v1,
        params_true_v2,
    )


class TestUnitcellMethods(unittest.TestCase):
    def test_unit_cell(self):
        out_dir = TemporaryDirectory()
        out_dir_name = out_dir.name + "/"

        v = np.array([[20.5, 0], [0, 28.8]])
        im = generate_crystal_image(v)[:, :220]
        params_true_v1 = np.array([0, 20.5])
        params_true_v2 = np.array([28.8, 0])
        v1, v2 = get_primitive_unit_cell_vectors(im, out_dir_name, "im", plot=False)
        print_vectors(v1, v2, params_true_v1, params_true_v2)
        param_true_list = [params_true_v1, params_true_v2]
        vector_list = [v1, v2]
        self.assertTrue(count_matches(param_true_list, vector_list, im.shape, 1e-3) == 2)

        v_2 = np.array([[35.6, 0], [0, 18.4]])
        im2 = generate_crystal_image(v_2)
        params_true_v1 = np.array([-18.4, 0])
        params_true_v2 = np.array([0, -35.6])
        param_true_list = [params_true_v1, params_true_v2]
        v1, v2 = get_primitive_unit_cell_vectors(im2, out_dir_name, "im2", plot=False)
        print_vectors(v1, v2, params_true_v1, params_true_v2)
        vector_list = [v1, v2]
        self.assertTrue(count_matches(param_true_list, vector_list, im2.shape, 1e-3) == 2)

        im3 = read_image("../examples/images/Nc3Nm.nc")[:210,]
        params_true_v1 = np.array([1.5, 25])
        params_true_v2 = np.array([50, 0])
        param_true_list = [params_true_v1, params_true_v2]
        v1, v2 = get_primitive_unit_cell_vectors(im3, out_dir_name, "im3", plot=False)
        print_vectors(v1, v2, params_true_v1, params_true_v2)
        vector_list = [v1, v2]
        self.assertTrue(count_matches(param_true_list, vector_list, im3.shape, 1e-3) == 2)

        im4 = read_image("../examples/images/bumps3.nc")
        params_true_v1 = np.array([0, 10])
        params_true_v2 = np.array([30, 0])
        param_true_list = [params_true_v1, params_true_v2]
        v1, v2 = get_primitive_unit_cell_vectors(im4, out_dir_name, "im4", plot=False)
        print_vectors(v1, v2, params_true_v1, params_true_v2)
        vector_list = [v1, v2]
        self.assertTrue(count_matches(param_true_list, vector_list, im4.shape, 1e-3) == 2)

        im5 = read_image("../examples/images/doubleSingle.nc")
        params_true_v1 = np.array([0, 15])
        params_true_v2 = np.array([25, 0])
        param_true_list = [params_true_v1, params_true_v2]
        v1, v2 = get_primitive_unit_cell_vectors(im5, out_dir_name, "im5", plot=False)
        print_vectors(v1, v2, params_true_v1, params_true_v2)
        vector_list = [v1, v2]
        self.assertTrue(count_matches(param_true_list, vector_list, im5.shape, 1e-3) == 2)

        im6 = read_image("../examples/images/hexVacancy.nc")
        params_true_v1 = np.array([12, 12])
        params_true_v2 = np.array([12, -12])
        param_true_list = [params_true_v1, params_true_v2]
        v1, v2 = get_primitive_unit_cell_vectors(im6, out_dir_name, "im6", plot=False, energy_exclusion_factor=1e32)
        print_vectors(v1, v2, params_true_v1, params_true_v2)
        vector_list = [v1, v2]
        self.assertTrue(count_matches(param_true_list, vector_list, im6.shape, 1e-3) == 2)

        im7 = read_image("../examples/images/left_grain.png", as_gray=True)
        params_true_v1 = np.array([11.4653, 14.2958])
        params_true_v2 = np.array([18.2864, -2.6899])
        param_true_list = [params_true_v1, params_true_v2]
        v1, v2 = get_primitive_unit_cell_vectors(im7, out_dir_name, "im7", plot=False)
        print_vectors(v1, v2, params_true_v1, params_true_v2)
        vector_list = [v1, v2]
        self.assertTrue(count_matches(param_true_list, vector_list, im7.shape, 1e-3) == 2)

        out_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
