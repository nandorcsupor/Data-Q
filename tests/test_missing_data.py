from data_class import DataClass


def test_wrong_data_types():
    data_class_wrong_data = DataClass("tests/test_data/missing_data.csv")
    assert data_class_wrong_data
    report = data_class_wrong_data.generate_report()
    missing_value_rows = report.get("MISSING_VALUE_ROWS")
    assert missing_value_rows
    assert missing_value_rows == [1, 2]


def test_correct_data_types():
    data_class_correct_data = DataClass("tests/test_data/correct_data_types.csv")
    assert data_class_correct_data
    report = data_class_correct_data.generate_report()
    assert not report.get("MISSING_VALUE_ROWS")
