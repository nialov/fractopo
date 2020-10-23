"""
Tests for config.py file
"""

from fractopo.analysis import config


class TestConfig:

    def test_get_color_dict(self):
        for unified in [True, False]:
            try:
                result = config.get_color_dict(unified=unified)
            except AssertionError:
                pass

        # Setup config stuff, then test
        # Number of target areas. Should be changed before analysis.
        config.n_ta = 4
        # Number of groups. Should be changed before analysis.
        config.n_g = 2
        # Target area name list
        config.ta_list = ["test_ta_1", "test_ta_2", "test_ta_3", "test_ta_4"]
        # Group name list
        config.g_list = ["test_g_1", "test_g_2"]

        for unified in [True, False]:
            result = config.get_color_dict(unified=unified)
            assert len(result) != 0

    def test_default_analysis_choices(self):
        # Test that all analyses are done by default
        assert all(list(config.choose_your_analyses.values()))

