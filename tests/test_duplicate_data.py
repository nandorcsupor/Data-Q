from data_class import DataClass


def test_duplicate_data():
    data_class_wrong_data = DataClass("tests/test_data/duplicate_data.csv")
    assert data_class_wrong_data
    report = data_class_wrong_data.generate_report()
    duplicates = report.get("DUPLICATE_ROWS")
    assert duplicates
    assert duplicates == [(1, 2)]


def test_no_duplicates():
    data_class_correct_data = DataClass("tests/test_data/correct_data_types.csv")
    assert data_class_correct_data
    report = data_class_correct_data.generate_report()
    assert not report.get("DUPLICATE_ROWS")
