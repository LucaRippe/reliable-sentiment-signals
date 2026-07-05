from data.config import DEFAULT_CONFIG_PATH, load_config


def test_load_config_has_data_section() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    assert "data" in config
    assert config["data"]["transcripts"]["dataset"] == "kurry/sp500_earnings_transcripts"
