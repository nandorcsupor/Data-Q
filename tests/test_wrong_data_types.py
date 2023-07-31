from data_class import DataClass


def test_wrong_data_types():
    data_class_wrong_data = DataClass("tests/test_data/wrong_data_types.csv")
    assert data_class_wrong_data
    report = data_class_wrong_data.generate_report()
    uniformity_report = report.get("UNIFORMITY")
    assert uniformity_report
    wrong_data_type_column = uniformity_report.get("CO2 Emissions(g/km)")
    assert wrong_data_type_column == [3]


def test_correct_data_types():
    data_class_correct_data = DataClass("tests/test_data/correct_data_types.csv")
    assert data_class_correct_data
    report = data_class_correct_data.generate_report()
    assert not report.get("UNIFORMITY")
