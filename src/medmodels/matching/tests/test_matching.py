import unittest
from medmodels.dataclass.dataclass import MedRecord
from medmodels.matching.matching import Matching


class TestMatching(unittest.TestCase):
    def setUp(self):
        self.treated_group = set({"P1", "P2"})
        self.control_group = set({"P3", "P4"})
        self.medrecord = MedRecord()

    def test_init(self):
        base_matching = Matching(self.medrecord, self.treated_group, self.control_group)
        self.assertEqual(base_matching.medrecord, self.medrecord)
        self.assertEqual(base_matching.treated_group, self.treated_group)
        self.assertEqual(base_matching.control_group, self.control_group)
        self.assertEqual(base_matching.matched_control, set())

        with self.assertRaises(AssertionError) as context:
            Matching(
                treated_group=set(),
                control_group=self.control_group,
                medrecord=self.medrecord,
            )
            self.assertTrue("Treated group cannot be empty" in str(context.exception))
        with self.assertRaises(AssertionError) as context:
            Matching(
                treated_group=self.treated_group,
                control_group=set(),
                medrecord=self.medrecord,
            )
            self.assertTrue("Control group cannot be empty" in str(context.exception))
        with self.assertRaises(AssertionError) as context:
            Matching(
                treated_group=self.treated_group,
                control_group=self.control_group,
                medrecord="MedRecord",
            )
            self.assertTrue(
                "medrecord must be a MedRecord object" in str(context.exception)
            )


if __name__ == "__main__":
    run_test = unittest.TestLoader().loadTestsFromTestCase(TestMatching)
    unittest.TextTestRunner(verbosity=2).run(run_test)
