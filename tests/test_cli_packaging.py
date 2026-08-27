from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import demo
import scaling

ROOT = Path(__file__).resolve().parents[1]


def test_readme_quotes_pip_extras_for_zsh():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "pip install -e '.[all]'" in readme
    assert "pip install -e .[all]" not in readme
    assert "durr-hoyer-demo" in readme
    assert "durr-hoyer-scale" in readme


def test_demo_exits_when_aer_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["demo.py"])
    monkeypatch.setitem(sys.modules, "qiskit_aer", types.ModuleType("qiskit_aer"))
    with pytest.raises(SystemExit) as exc:
        demo.main()
    assert exc.value.code == 1
    assert "qiskit-aer" in capsys.readouterr().err


def _hide_plotting(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("closest_search.plotting")
    monkeypatch.setitem(sys.modules, "closest_search.plotting", fake)
    import closest_search

    monkeypatch.setattr(closest_search, "plotting", fake, raising=False)


def test_demo_exits_when_plotting_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["demo.py", "--save-plots"])
    monkeypatch.setattr(demo, "histogram_of_closest_set", lambda *a, **k: {0: 1})
    _hide_plotting(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        demo.main()
    assert exc.value.code == 1
    assert "matplotlib" in capsys.readouterr().err


def test_scaling_exits_when_plotting_is_missing(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["scaling.py", "--max-n", "2", "--save-plots"])
    _hide_plotting(monkeypatch)
    with pytest.raises(SystemExit) as exc:
        scaling.main()
    assert exc.value.code == 1
    assert "matplotlib" in capsys.readouterr().err
