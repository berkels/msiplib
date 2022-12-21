import unittest
from shutil import rmtree
from sphinx.application import Sphinx


class DocTest(unittest.TestCase):

    def setUp(self):
        self.output_dir = '../docs/build_test'

    def test_html_documentation(self):
        app = Sphinx(srcdir='../docs',
                     confdir='../docs',
                     outdir=self.output_dir,
                     doctreedir='../docs/build_test/doctrees',
                     buildername='html',
                     warningiserror=True)
        app.build(force_all=True)

    def tearDown(self):
        rmtree(self.output_dir)


if __name__ == '__main__':
    unittest.main()
