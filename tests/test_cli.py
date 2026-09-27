from click.testing import CliRunner

from inspect_boltons.cli import main


def test_help():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
