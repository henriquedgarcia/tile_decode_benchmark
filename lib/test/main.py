import unittest
from util import update_dictionary, AutoDict, dishevel_dictionary


class TestUtil(unittest.TestCase):
    def test_update_dictionary(self):
        source = AutoDict()
        update_dictionary(50, source, 'bola', 'carro', 'casa', 'coisa', 'antes')
        reference = AutoDict({
            'bola': {
                'carro': {
                    'casa': {
                        'coisa': {
                            'antes': {
                                'nova': 50
                            }}}}}})

        self.assertEqual(source, reference)

    def test_dishevel_dictionary(self):
        source = AutoDict({
            'bola': {
                'carro': {
                    'casa': {
                        'coisa': {
                            'antes': {
                                'nova': 50
                            }}}}}})
        dishevel_dictionary(source, 'bola', 'carro', 'coisa')
        reference = AutoDict({'antes': {'nova': 50}})
        self.assertEqual(source, reference)


if __name__ == '__main__':
    unittest.main()
