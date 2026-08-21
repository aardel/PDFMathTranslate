import unittest

from pdf2zh.pdfinterp import pdf_number


class TestPdfNumber(unittest.TestCase):
    def test_small_matrix_values_use_fixed_point_pdf_syntax(self):
        values = (
            2.638613030907712e-07,
            8.772340414052807e-08,
            1.5748217498632033,
            -0.0,
        )

        rendered = [pdf_number(value) for value in values]

        self.assertEqual(rendered[0], "0.000000263861303")
        self.assertEqual(rendered[1], "0.000000087723404")
        self.assertEqual(rendered[2], "1.574821749863203")
        self.assertEqual(rendered[3], "0")
        self.assertTrue(all("e" not in value.lower() for value in rendered))


if __name__ == "__main__":
    unittest.main()
