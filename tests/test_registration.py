import numpy as np
import unittest
from msiplib.registration import ParametricAffineDeformation, parametric_registration_mld


class TestRegistrationMethods(unittest.TestCase):

    def test_affine_registration(self):
        inputimage1 = np.zeros((64, 64))
        inputimage1[10:30, 15:45] = 1
        params_true = np.array([0.1, -0.1, 0.9, 0.1, 0.1, 0.9])

        def_type = ParametricAffineDeformation
        phi = def_type(params_true, inputimage1.shape)
        inputimage2 = phi.deform_image(inputimage1)

        params = parametric_registration_mld(inputimage1, inputimage2, def_type, 4, minimizer='trf')[2]
        self.assertTrue(np.linalg.norm(params_true - params) <= 1e-10)

        params = parametric_registration_mld(inputimage1, inputimage2, def_type, 4, minimizer='GaussNewton')[2]
        self.assertTrue(np.linalg.norm(params_true - params) <= 1e-8)

        params = parametric_registration_mld(inputimage1, inputimage2, def_type, 4, minimizer='L-BFGS-B')[2]
        self.assertTrue(np.linalg.norm(params_true - params) <= 1e-4)


if __name__ == '__main__':
    unittest.main()
