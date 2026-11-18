import trialcost


def test_package_imports():
    assert trialcost.__version__


def test_data_dir_has_catalogue():
    expected = {"procedures.csv", "study_items.csv", "countries.csv", "benchmarks.csv"}
    assert expected <= {p.name for p in trialcost.DATA_DIR.glob("*.csv")}
