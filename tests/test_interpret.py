"""Unit tests for eva.interpret."""

from __future__ import annotations

import numpy as np
import pytest

SFREQ = 250.0
N_CH = 3
N_SAMP = 500
N_EPOCHS = 6


@pytest.fixture
def synced_h5(tmp_path):
    """Minimal .h5 with /eeg and /behavioral groups, as produced by preprocess()+sync()."""
    import h5py

    rng = np.random.default_rng(0)
    data = rng.standard_normal((N_EPOCHS, N_CH, N_SAMP)).astype(np.float32) * 20e-6
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)  # 2 conditions, 3 epochs each
    scores = np.linspace(0.2, 0.9, N_EPOCHS)

    path = tmp_path / "subject.h5"
    with h5py.File(path, "w") as f:
        eeg = f.create_group("eeg")
        eeg.create_dataset("data", data=data)
        eeg.create_dataset("labels", data=labels)
        eeg.create_dataset("ch_names", data=np.array(["Fz", "Cz", "Pz"], dtype=object))
        eeg.create_dataset("label_names", data=np.array(["cond_a", "cond_b"], dtype=object))
        eeg.create_dataset("label_codes", data=np.array([0, 1], dtype=np.int32))
        meta = f.create_group("metadata")
        meta.attrs["sfreq"] = np.float32(SFREQ)
        meta.attrs["tmin"] = np.float32(0.0)
        meta.attrs["tmax"] = np.float32(2.0)
        beh = f.create_group("behavioral")
        beh.create_dataset("score", data=scores.astype(np.float32))
    return path


# ---------------------------------------------------------------------------
# interpret() — validation
# ---------------------------------------------------------------------------

class TestInterpretValidation:
    def test_missing_file_raises(self, tmp_path):
        from eva import interpret
        with pytest.raises(FileNotFoundError):
            interpret(tmp_path / "ghost.h5")

    def test_no_eeg_group_raises(self, tmp_path):
        import h5py
        from eva import interpret
        path = tmp_path / "empty.h5"
        with h5py.File(path, "w") as f:
            f.create_group("other")
        with pytest.raises(ValueError, match="no '/eeg/'"):
            interpret(path)

    def test_missing_behavioral_group_raises(self, tmp_path):
        import h5py
        from eva import interpret
        path = tmp_path / "no_behavioral.h5"
        with h5py.File(path, "w") as f:
            eeg = f.create_group("eeg")
            eeg.create_dataset("data", data=np.zeros((2, 2, 100), dtype=np.float32))
            meta = f.create_group("metadata")
            meta.attrs["sfreq"] = np.float32(SFREQ)
        with pytest.raises(ValueError, match="behavioral"):
            interpret(path)

    def test_missing_score_key_raises(self, synced_h5):
        from eva import interpret
        with pytest.raises(ValueError, match="behavioral/accuracy"):
            interpret(synced_h5, score_key="accuracy")

    def test_band_exceeding_nyquist_raises(self, synced_h5):
        from eva import interpret
        with pytest.raises(ValueError, match="Nyquist"):
            interpret(synced_h5, bands={"custom": (200.0, 300.0)})

    def test_inverted_band_raises(self, synced_h5):
        from eva import interpret
        with pytest.raises(ValueError, match="Invalid frequency band"):
            interpret(synced_h5, bands={"custom": (8.0, 4.0)})

    def test_band_name_in_error_message(self, synced_h5):
        from eva import interpret
        with pytest.raises(ValueError, match="'gamma'"):
            interpret(synced_h5, bands={"theta": (4.0, 8.0), "gamma": (30.0, 200.0)})


# ---------------------------------------------------------------------------
# interpret() — output generation
# ---------------------------------------------------------------------------

class TestInterpretOutput:
    def test_generates_html_and_csv(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5)
        assert out.exists()
        assert out.name == "subject_results.html"
        assert (out.parent / "subject_summary.csv").exists()

    def test_default_output_dir(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5)
        assert out.parent == synced_h5.parent / "results"

    def test_custom_output_dir_respected(self, synced_h5, tmp_path):
        from eva import interpret
        custom_dir = tmp_path / "custom_results"
        out = interpret(synced_h5, output_dir=custom_dir)
        assert out.parent == custom_dir

    def test_label_names_mapping_applied(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5, label_names={"cond_a": "Task A", "cond_b": "Task B"})
        html = out.read_text(encoding="utf-8")
        assert "Task A" in html
        assert "Task B" in html

    def test_raw_label_shown_when_no_mapping(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5)
        html = out.read_text(encoding="utf-8")
        assert "cond_a" in html
        assert "cond_b" in html

    def test_disclaimer_present(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5)
        html = out.read_text(encoding="utf-8")
        assert "descriptive only" in html
        assert "not a diagnostic tool" in html

    def test_default_bands_all_shown(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5)
        html = out.read_text(encoding="utf-8")
        assert "Theta activity" in html
        assert "Alpha activity" in html
        assert "Beta activity" in html

    def test_custom_bands_shown_and_defaults_absent(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5, bands={"custom": (4.0, 10.0)})
        html = out.read_text(encoding="utf-8")
        assert "Custom activity" in html
        assert "Beta activity" not in html

    def test_single_band_still_works(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5, bands={"theta": (4.0, 8.0)})
        html = out.read_text(encoding="utf-8")
        assert "Theta activity" in html
        assert "Alpha activity" not in html


class TestGlossary:
    def test_default_bands_have_explanations_and_links(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5)
        html = out.read_text(encoding="utf-8")
        assert "What Do These Bands Mean?" in html
        assert 'href="https://pubmed.ncbi.nlm.nih.gov/38723734/"' in html
        assert 'target="_blank"' in html
        assert "No established functional interpretation" not in html

    def test_custom_band_gets_neutral_fallback_not_invented_claim(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5, bands={"custom": (4.0, 10.0)})
        html = out.read_text(encoding="utf-8")
        assert "No established functional interpretation" in html

    def test_glossary_only_covers_bands_actually_used(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5, bands={"theta": (4.0, 8.0)})
        html = out.read_text(encoding="utf-8")
        assert "<h3>Theta</h3>" in html
        assert "<h3>Alpha</h3>" not in html
        assert "<h3>Beta</h3>" not in html

    def test_glossary_html_helper_unknown_band(self):
        from eva.interpret import _glossary_html
        html = _glossary_html(["not_a_real_band"], "en")
        assert "No established functional interpretation" in html
        assert "<h3>Not_a_real_band</h3>" in html


class TestCalculationExplanation:
    def test_how_calculated_note_present(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5)
        html = out.read_text(encoding="utf-8")
        assert "How are these numbers calculated?" in html


class TestClassifyElectrode:
    @pytest.mark.parametrize("name,expected", [
        ("Fz", "Frontal"),
        ("FC1", "Fronto-Central"),
        ("FC2", "Fronto-Central"),
        ("Cz", "Central"),
        ("C3", "Central"),
        ("C4", "Central"),
        ("Pz", "Parietal"),
        ("P4", "Parietal"),
        ("Fp1", "Frontopolar"),
        ("AF3", "Anterior Frontal"),
        ("CP5", "Centro-Parietal"),
        ("TP9", "Temporo-Parietal"),
        ("PO7", "Parieto-Occipital"),
        ("T7", "Temporal"),
        ("O2", "Occipital"),
        ("M1", "Mastoid"),
        ("A2", "Mastoid/Ear reference"),
    ])
    def test_known_10_20_names(self, name, expected):
        from eva.interpret import _classify_electrode
        assert _classify_electrode(name) == expected

    @pytest.mark.parametrize("name", ["CH1", "EEG_04", "REF", "GND", "EOG", "ECG", ""])
    def test_unrecognized_names_return_none(self, name):
        from eva.interpret import _classify_electrode
        assert _classify_electrode(name) is None

    def test_case_insensitive(self):
        from eva.interpret import _classify_electrode
        assert _classify_electrode("fz") == "Frontal"
        assert _classify_electrode("fc1") == "Fronto-Central"


class TestElectrodeLegend:
    def test_groups_by_region_in_fixed_order(self):
        from eva.interpret import _electrode_legend_html
        html = _electrode_legend_html(["Cz", "Fz", "FC1", "Pz"], "en")
        # Frontal appears before Fronto-Central before Central before Parietal
        # per the _REGION_PREFIXES order, regardless of input order.
        assert html.index("Frontal:") < html.index("Fronto-Central:")
        assert html.index("Fronto-Central:") < html.index("Central:")
        assert html.index("Central:") < html.index("Parietal:")

    def test_unclassified_channels_listed_separately(self):
        from eva.interpret import _electrode_legend_html
        html = _electrode_legend_html(["Fz", "CH9"], "en")
        assert "Frontal: Fz" in html
        assert "Not classified: CH9" in html

    def test_all_unclassified(self):
        from eva.interpret import _electrode_legend_html
        html = _electrode_legend_html(["CH1", "CH2"], "en")
        assert html == "Not classified: CH1, CH2"

    def test_pt_br_translates_region_names(self):
        from eva.interpret import _electrode_legend_html
        html = _electrode_legend_html(["Fz", "Cz", "CH9"], "pt-br")
        assert "Frontal: Fz" in html
        assert "Central: Cz" in html
        assert "Não classificado: CH9" in html


class TestSummarizePerChannel:
    def test_shape_and_grouping(self):
        from eva.interpret import _summarize_per_channel
        conditions = np.array(["a", "a", "b"])
        ch_names = ["Fz", "Cz"]
        band_activity_by_channel = {
            "theta": np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 20.0]]),
        }
        result = _summarize_per_channel(conditions, ch_names, band_activity_by_channel, None)
        assert set(result.keys()) == {"theta"}
        table = result["theta"]
        assert list(table.columns) == ["Fz", "Cz"]
        assert table.loc["a", "Fz"] == pytest.approx(2.0)   # mean(1.0, 3.0)
        assert table.loc["a", "Cz"] == pytest.approx(3.0)   # mean(2.0, 4.0)
        assert table.loc["b", "Fz"] == pytest.approx(10.0)

    def test_label_mapping_consolidates_rows(self):
        from eva.interpret import _summarize_per_channel
        conditions = np.array(["vr_mem8", "vr_mem9"])
        ch_names = ["Fz"]
        band_activity_by_channel = {"theta": np.array([[1.0], [3.0]])}
        result = _summarize_per_channel(
            conditions, ch_names, band_activity_by_channel,
            {"vr_mem8": "Memory", "vr_mem9": "Memory"},
        )
        table = result["theta"]
        assert list(table.index) == ["Memory"]
        assert table.loc["Memory", "Fz"] == pytest.approx(2.0)


class TestPerElectrodeReport:
    def test_electrode_table_present_for_each_band(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5, bands={"theta": (4.0, 8.0), "alpha": (8.0, 13.0)})
        html = out.read_text(encoding="utf-8")
        assert "Theta power by electrode" in html
        assert "Alpha power by electrode" in html

    def test_electrode_columns_present(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5)
        html = out.read_text(encoding="utf-8")
        assert "<th>Fz</th>" in html
        assert "<th>Cz</th>" in html
        assert "<th>Pz</th>" in html

    def test_electrode_legend_in_report(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5)
        html = out.read_text(encoding="utf-8")
        assert "Frontal: Fz" in html
        assert "Central: Cz" in html
        assert "Parietal: Pz" in html


class TestLanguage:
    def test_default_is_english(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5)
        assert out.name == "subject_results.html"
        html = out.read_text(encoding="utf-8")
        assert '<html lang="en">' in html
        assert "Results Summary" in html

    def test_invalid_language_raises(self, synced_h5):
        from eva import interpret
        with pytest.raises(ValueError, match="Unsupported language"):
            interpret(synced_h5, language="fr")

    def test_invalid_language_raises_before_touching_file(self, tmp_path):
        # Fails fast on a bad language even for a file that doesn't exist,
        # confirming the check happens before any I/O.
        from eva import interpret
        with pytest.raises(ValueError, match="Unsupported language"):
            interpret(tmp_path / "ghost.h5", language="fr")

    def test_pt_br_output_filename(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5, language="pt-br")
        assert out.name == "subject_results_pt-br.html"

    def test_pt_br_html_lang_attribute(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5, language="pt-br")
        html = out.read_text(encoding="utf-8")
        assert '<html lang="pt-BR">' in html

    def test_pt_br_translates_headings_and_labels(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5, language="pt-br")
        html = out.read_text(encoding="utf-8")
        assert "Resumo de Resultados" in html
        assert "Por Tarefa / Condição" in html
        assert "O Que Essas Bandas Significam?" in html
        assert "Detalhes técnicos" in html
        assert "Pontuação cognitiva" in html
        assert "diagnóstica" in html

    def test_pt_br_translates_band_display_names(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5, language="pt-br")
        html = out.read_text(encoding="utf-8")
        assert "Atividade Teta" in html
        assert "Atividade Alfa" in html
        assert "Atividade Beta" in html

    def test_pt_br_glossary_translated(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5, language="pt-br")
        html = out.read_text(encoding="utf-8")
        assert "memória de trabalho" in html  # from the theta explanation

    def test_pt_br_custom_band_fallback_translated(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5, bands={"custom": (4.0, 10.0)}, language="pt-br")
        html = out.read_text(encoding="utf-8")
        assert "Não há interpretação funcional estabelecida" in html

    def test_pt_br_references_stay_in_english(self, synced_h5):
        # Academic reference titles are never translated, in either language.
        from eva import interpret
        out = interpret(synced_h5, language="pt-br")
        html = out.read_text(encoding="utf-8")
        assert "Modulation of human frontal midline theta by neurofeedback" in html

    def test_pt_br_epoch_pluralization(self, synced_h5):
        from eva import interpret
        out = interpret(synced_h5)  # 3 epochs for cond_a, matches synced_h5 fixture
        html_en = out.read_text(encoding="utf-8")
        assert "3 epochs" in html_en

        out_pt = interpret(synced_h5, language="pt-br")
        html_pt = out_pt.read_text(encoding="utf-8")
        assert "3 épocas" in html_pt

    def test_english_output_unaffected_by_language_feature(self, synced_h5):
        # Default English report keeps its original filename/content shape.
        from eva import interpret
        out = interpret(synced_h5)
        html = out.read_text(encoding="utf-8")
        assert "How are these numbers calculated?" in html
        assert "Cognitive score" in html


class TestBandDisplayName:
    def test_known_band_translated(self):
        from eva.interpret import _band_display_name
        assert _band_display_name("theta", "en") == "Theta"
        assert _band_display_name("theta", "pt-br") == "Teta"

    def test_unknown_band_falls_back_to_capitalized_key(self):
        from eva.interpret import _band_display_name
        assert _band_display_name("custom", "en") == "Custom"
        assert _band_display_name("custom", "pt-br") == "Custom"


class TestEpochCountLabel:
    def test_english_singular_plural(self):
        from eva.interpret import _epoch_count_label
        assert _epoch_count_label(1, "en") == "1 epoch"
        assert _epoch_count_label(2, "en") == "2 epochs"

    def test_portuguese_singular_plural(self):
        from eva.interpret import _epoch_count_label
        assert _epoch_count_label(1, "pt-br") == "1 época"
        assert _epoch_count_label(2, "pt-br") == "2 épocas"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class TestBandPowerPerEpoch:
    def test_output_keys_and_shape(self, synced_h5):
        import h5py
        from eva.interpret import _band_power_per_epoch
        with h5py.File(synced_h5, "r") as f:
            data = f["eeg/data"][:]
        power = _band_power_per_epoch(data, SFREQ, {"theta": (4.0, 8.0), "alpha": (8.0, 13.0)})
        assert set(power.keys()) == {"theta", "alpha"}
        assert power["theta"].shape == (N_EPOCHS,)
        assert np.all(np.isfinite(power["theta"]))

    def test_nonoverlapping_bands_differ(self):
        from eva.interpret import _band_power_per_epoch
        # Deterministic 6 Hz tone (theta band) — theta power must dominate
        # beta power, unlike white noise where band power is similar by chance.
        t = np.arange(N_SAMP) / SFREQ
        tone = np.sin(2 * np.pi * 6.0 * t).astype(np.float32) * 20e-6
        data = np.tile(tone, (N_EPOCHS, N_CH, 1))
        power = _band_power_per_epoch(data, SFREQ, {"theta": (4.0, 8.0), "beta": (13.0, 30.0)})
        assert np.all(power["theta"] > power["beta"] * 10)


class TestMinMax0100:
    def test_full_range(self):
        from eva.interpret import _minmax_0_100
        result = _minmax_0_100(np.array([0.0, 5.0, 10.0]))
        np.testing.assert_allclose(result, [0.0, 50.0, 100.0])

    def test_constant_input_returns_midpoint(self):
        from eva.interpret import _minmax_0_100
        result = _minmax_0_100(np.array([5.0, 5.0, 5.0]))
        np.testing.assert_allclose(result, [50.0, 50.0, 50.0])


class TestSummarize:
    def test_one_row_per_condition(self):
        from eva.interpret import _summarize
        conditions = np.array(["a", "a", "a", "b", "b", "b"])
        scores = np.linspace(0.2, 0.9, 6)
        band_activity = {"theta": np.linspace(1e-12, 2e-12, 6)}
        summary = _summarize(conditions, scores, band_activity, None)
        assert len(summary) == 2
        assert set(summary["condition"]) == {"a", "b"}
        assert set(summary["n_epochs"]) == {3}
        assert "theta_raw" in summary.columns
        assert "theta_level" in summary.columns

    def test_condition_order_preserved(self):
        from eva.interpret import _summarize
        conditions = np.array(["b", "b", "a", "a"])
        scores = np.array([1.0, 1.0, 2.0, 2.0])
        band_activity = {"theta": np.array([1.0, 1.0, 1.0, 1.0])}
        summary = _summarize(conditions, scores, band_activity, None)
        assert summary["condition"].tolist() == ["b", "a"]

    def test_label_mapping_applied(self):
        from eva.interpret import _summarize
        conditions = np.array(["vr_att", "vr_att"])
        scores = np.array([0.5, 0.7])
        band_activity = {"theta": np.array([1.0, 1.2])}
        summary = _summarize(conditions, scores, band_activity, {"vr_att": "Attention"})
        assert summary["condition"].tolist() == ["Attention"]
        assert summary["condition_raw"].tolist() == ["vr_att"]

    def test_unmapped_condition_keeps_raw_label(self):
        from eva.interpret import _summarize
        conditions = np.array(["vr_att", "vr_abs"])
        scores = np.array([0.5, 0.7])
        band_activity = {"theta": np.array([1.0, 1.2])}
        summary = _summarize(conditions, scores, band_activity, {"vr_att": "Attention"})
        row = summary.set_index("condition_raw")
        assert row.loc["vr_att", "condition"] == "Attention"
        assert row.loc["vr_abs", "condition"] == "vr_abs"

    def test_multiple_raw_labels_consolidated_under_shared_display_name(self):
        from eva.interpret import _summarize
        conditions = np.array(["vr_mem8", "vr_mem9", "vr_mem10", "vr_att"])
        scores = np.array([0.5, 0.6, 0.7, 0.9])
        band_activity = {"theta": np.array([1.0, 1.1, 1.2, 2.0])}
        mapping = {"vr_mem8": "Memory", "vr_mem9": "Memory", "vr_mem10": "Memory",
                   "vr_att": "Attention"}
        summary = _summarize(conditions, scores, band_activity, mapping)
        assert len(summary) == 2
        row = summary.set_index("condition")
        assert row.loc["Memory", "n_epochs"] == 3
        assert row.loc["Memory", "score_raw"] == pytest.approx(0.6)
        assert row.loc["Attention", "n_epochs"] == 1

    def test_multiple_bands_produce_independent_columns(self):
        from eva.interpret import _summarize
        conditions = np.array(["a", "a"])
        scores = np.array([0.5, 0.7])
        band_activity = {
            "theta": np.array([1.0, 3.0]),
            "beta": np.array([9.0, 7.0]),
        }
        summary = _summarize(conditions, scores, band_activity, None)
        row = summary.iloc[0]
        assert row["theta_raw"] == pytest.approx(2.0)
        assert row["beta_raw"] == pytest.approx(8.0)
