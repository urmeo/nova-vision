import csv
import json

import pytest
from PIL import Image

from novavision.eval import human_study
from novavision.eval.metrics import cohen_kappa
from novavision.experiments import run
from novavision.experiments.manifest import build_manifest
from novavision.generation.base import NullBackend
from novavision.prompting import NEGATIVE_PROMPT, build_prompt
from novavision.taxonomy import EMOTIONS


def _read_csv(path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def _replayable_payload(records, width=64, height=64):
    cfg = {
        "backend": "null",
        "diffusion_model": "stabilityai/sd-turbo",
        "style": "artistic",
        "base_seed": 0,
        "width": width,
        "height": height,
    }
    manifest = build_manifest(**cfg)
    manifest["source_sha256"] = run._source_fingerprint()
    for r in records:
        idx = r.get("index", 0)
        image = NullBackend().generate(
            build_prompt(
                r["content"],
                emotion=r["intended"],
                valence=r["intended_valence"],
                arousal=r["intended_arousal"],
                style="artistic",
                tier=r["tier"],
            ),
            width=width,
            height=height,
            seed=idx * 97 + EMOTIONS.index(r["intended"]) * 13 + r["seed"],
            negative_prompt=NEGATIVE_PROMPT,
        )
        r["image_pixel_sha256"] = run._image_digest(image)
    return {"manifest": manifest, "records": records}


def test_cohen_kappa_perfect_and_chance():
    assert cohen_kappa(["joy", "anger"], ["joy", "anger"], EMOTIONS) == 1.0

    assert cohen_kappa(["joy", "anger"], ["anger", "joy"], EMOTIONS) < 0


def _fake_results(tmp_path):
    records = []
    for sk in range(2):
        for e in EMOTIONS:
            records.append(
                {
                    "probe": "clip:test",
                    "tier": "affect",
                    "content": "a city street with buildings",
                    "intended": e,
                    "seed": sk,
                    "predicted": e,
                    "intended_valence": 0.0,
                    "intended_arousal": 0.5,
                    "recovered_valence": 0.0,
                    "recovered_arousal": 0.5,
                    "clip_t": 0.2,
                }
            )
    payload = _replayable_payload(records)
    out = tmp_path / "results.json"
    out.write_text(json.dumps(payload))
    return tmp_path


def test_build_sheet_and_analyze(tmp_path):
    results_dir = _fake_results(tmp_path)
    study = human_study.build_sheet(results_dir, n=7, seed=0)
    assert (study / "ratings_template.csv").exists()
    assert (study / "key.csv").exists()
    assert len(list((study / "images").glob("*.png"))) == 7

    key = _read_csv(study / "key.csv")
    template = _read_csv(study / "ratings_template.csv")
    assert all(set(record) == {*human_study._IDENTITY_FIELDS, "emotion"} for record in template)
    rated = study / "rated.csv"
    human_study._write_csv(
        rated,
        [*human_study._IDENTITY_FIELDS, "emotion"],
        [
            {**record, "emotion": target["probe"]}
            for record, target in zip(template, key, strict=True)
        ],
    )
    result = human_study.analyze(rated, study / "key.csv")
    assert result["n_rated"] == 7
    assert result["human_vs_probe_kappa"] == 1.0


def _text_record(with_index: bool) -> dict:
    r = {
        "tier": "emotion",
        "content": "a terrible storm hit the village",
        "intended": "fear",
        "seed": 0,
        "predicted": "fear",
        "intended_valence": -0.7,
        "intended_arousal": 0.8,
        "recovered_valence": -0.5,
        "recovered_arousal": 0.6,
        "clip_t": 0.2,
    }
    if with_index:
        r["index"] = 3
    return r


def _text_payload(record) -> dict:
    return _replayable_payload([record], width=32, height=32)


def test_build_sheet_text_track_with_index(tmp_path):

    (tmp_path / "results.json").write_text(json.dumps(_text_payload(_text_record(with_index=True))))
    study = human_study.build_sheet(tmp_path, n=1, seed=0)
    assert len(list((study / "images").glob("*.png"))) == 1


def test_build_sheet_rejects_unreproducible_record(tmp_path):

    (tmp_path / "results.json").write_text(
        json.dumps(_text_payload(_text_record(with_index=False)))
    )
    with pytest.raises(ValueError, match="Cannot reproduce"):
        human_study.build_sheet(tmp_path, n=1, seed=0)


def test_analyze_skips_out_of_vocab_rating(tmp_path):
    key = tmp_path / "key.csv"
    human_study._write_csv(
        key,
        [*human_study._IDENTITY_FIELDS, "intended", "probe"],
        [
            {**_identity(0), "intended": "joy", "probe": "joy"},
            {**_identity(1), "intended": "anger", "probe": "anger"},
        ],
    )
    rated = tmp_path / "rated.csv"

    human_study._write_csv(
        rated,
        [*human_study._IDENTITY_FIELDS, "emotion"],
        [
            {**_identity(0), "emotion": "happy"},
            {**_identity(1), "emotion": "banana"},
        ],
    )
    res = human_study.analyze(rated, key)
    assert res["n_rated"] == 1 and res["n_unscored"] == 1
    assert res["unscored_ids"] == [_identity(1)["id"]]


def _identity(index, study_id="1" * 32):
    image_id = f"{study_id}-{index:03d}"
    return {
        "id": image_id,
        "study_id": study_id,
        "image": f"images/{image_id}.png",
        "image_pixel_sha256": "a" * 64,
    }


def _write_study_files(tmp_path, ratings, key):
    ratings_path, key_path = tmp_path / "ratings.csv", tmp_path / "key.csv"
    human_study._write_csv(ratings_path, [*human_study._IDENTITY_FIELDS, "emotion"], ratings)
    human_study._write_csv(key_path, [*human_study._IDENTITY_FIELDS, "intended", "probe"], key)
    return ratings_path, key_path


@pytest.mark.parametrize("duplicate_file", ["ratings", "key"])
def test_analyze_rejects_duplicate_image_ids(tmp_path, duplicate_file):
    ratings = [{**_identity(0), "emotion": "joy"}]
    key = [{**_identity(0), "intended": "joy", "probe": "joy"}]
    (ratings if duplicate_file == "ratings" else key).append(
        {**_identity(0), "emotion": "anger"} if duplicate_file == "ratings" else key[0]
    )
    ratings_path, key_path = _write_study_files(tmp_path, ratings, key)
    with pytest.raises(ValueError, match=f"Duplicate image ID {_identity(0)['id']}"):
        human_study.analyze(ratings_path, key_path)


def test_analyze_rejects_rated_image_absent_from_the_key(tmp_path):
    ratings_path, key_path = _write_study_files(
        tmp_path,
        [{**_identity(1), "emotion": "joy"}],
        [{**_identity(0), "intended": "joy", "probe": "joy"}],
    )
    with pytest.raises(ValueError, match="image IDs absent from the key"):
        human_study.analyze(ratings_path, key_path)


def test_build_sheet_refuses_nondeterministic_backend(tmp_path):
    d = _fake_results(tmp_path)
    payload = json.loads((d / "results.json").read_text())
    payload["manifest"]["config"]["backend"] = "hf-api"
    (d / "results.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="non-deterministic"):
        human_study.build_sheet(d, n=4, seed=0)


def test_build_sheet_reports_per_class_counts(tmp_path):
    study = human_study.build_sheet(_fake_results(tmp_path), n=4, seed=0)
    counts = _read_csv(study / "counts.csv")
    assert {r["emotion"] for r in counts} == set(EMOTIONS)
    assert sum(int(r["n"]) for r in counts) == 4
    assert not (study / "README.md").exists()


def test_stratified_sample_hides_canonical_emotion_order_and_is_repeatable():
    records = [
        {"tier": "affect", "intended": emotion, "seed": seed}
        for seed in range(2)
        for emotion in EMOTIONS
    ]
    sample = human_study._sample_records(records, n=14, seed=0)
    assert sample == human_study._sample_records(records, n=14, seed=0)
    assert sorted(record["intended"] for record in sample) == sorted(list(EMOTIONS) * 2)
    assert [record["intended"] for record in sample] != list(EMOTIONS) * 2


def test_missing_pixel_digest_fails_before_backend_initialization(tmp_path, monkeypatch):
    payload = {
        "manifest": {"config": {"backend": "null"}},
        "records": [_text_record(with_index=True)],
    }
    (tmp_path / "results.json").write_text(json.dumps(payload))
    monkeypatch.setattr(
        human_study, "get_backend", lambda *a, **kw: pytest.fail("backend initialized")
    )
    with pytest.raises(ValueError, match="missing pixel digests"):
        human_study.build_sheet(tmp_path)


@pytest.mark.parametrize("changed", ["source_sha256", "python", "packages"])
def test_build_sheet_refuses_changed_provenance_before_loading(tmp_path, monkeypatch, changed):
    _fake_results(tmp_path)
    path = tmp_path / "results.json"
    payload = json.loads(path.read_text())
    payload["manifest"][changed] = {} if changed == "packages" else "different"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(
        human_study, "get_backend", lambda *a, **kw: pytest.fail("backend initialized")
    )
    with pytest.raises(ValueError, match="provenance differs"):
        human_study.build_sheet(tmp_path)


def test_replay_allows_new_head_when_source_and_environment_match(tmp_path, monkeypatch):
    from novavision.experiments import manifest

    _fake_results(tmp_path)
    original = manifest.build_manifest

    def later_commit(**cfg):
        current = original(**cfg)
        current["git_sha"] = "a" * 40
        return current

    monkeypatch.setattr(manifest, "build_manifest", later_commit)
    assert (human_study.build_sheet(tmp_path, n=1) / "ratings_template.csv").exists()


def test_replay_still_requires_known_recorded_git_sha(tmp_path):
    _fake_results(tmp_path)
    path = tmp_path / "results.json"
    payload = json.loads(path.read_text())
    payload["manifest"]["git_sha"] = "unknown"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="provenance differs"):
        human_study.build_sheet(tmp_path)


def _saved_original_payload(tmp_path):
    image = Image.new("RGB", (8, 8), "blue")
    image.save(tmp_path / "original.png")
    record = _text_record(with_index=False)
    record.update(image_path="original.png", image_pixel_sha256=run._image_digest(image))

    return {"manifest": {"config": {"backend": "hf-api"}}, "records": [record]}


def test_build_sheet_uses_verified_original_and_custom_destination(tmp_path, monkeypatch):
    (tmp_path / "results.json").write_text(json.dumps(_saved_original_payload(tmp_path)))
    monkeypatch.setattr(
        human_study, "get_backend", lambda *a, **kw: pytest.fail("backend initialized")
    )
    destination = tmp_path / "separate" / "study"
    study = human_study.build_sheet(tmp_path, n=1, out=destination)
    assert study == destination
    image_path = study / _read_csv(study / "key.csv")[0]["image"]
    assert run._image_digest(Image.open(image_path)) == run._image_digest(
        Image.open(tmp_path / "original.png")
    )
    assert not (tmp_path / "human_study").exists()


def test_saved_original_with_wrong_digest_is_rejected(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps(_saved_original_payload(tmp_path)))
    Image.new("RGB", (8, 8), "red").save(tmp_path / "original.png")
    with pytest.raises(ValueError, match="pixel digest differs"):
        human_study.build_sheet(tmp_path, n=1)


def test_supplied_backend_does_not_bypass_missing_provenance(tmp_path):
    _fake_results(tmp_path)
    path = tmp_path / "results.json"
    payload = json.loads(path.read_text())
    del payload["manifest"]["source_sha256"]
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="incomplete provenance"):
        human_study.build_sheet(tmp_path, gen=NullBackend())


def test_rebuilt_image_must_match_recorded_pixels(tmp_path):
    _fake_results(tmp_path)

    class WrongGenerator:
        def generate(self, *a, **kw):
            return Image.new("RGB", (64, 64), "red")

    with pytest.raises(ValueError, match="pixel digest differs"):
        human_study.build_sheet(tmp_path, gen=WrongGenerator())


def test_diffusers_replay_uses_recorded_generator_settings(tmp_path, monkeypatch):
    _fake_results(tmp_path)
    path = tmp_path / "results.json"
    payload = json.loads(path.read_text())
    cfg = payload["manifest"]["config"]
    cfg.update(backend="diffusers", device="cpu", dtype="float32", generation_steps=2)
    path.write_text(json.dumps(payload))
    revision = payload["manifest"]["model_revisions"]["diffusion"]
    calls = []

    class SavedSettingsGenerator(NullBackend):
        device = "cpu"
        dtype = "float32"
        steps = 2
        model_id = cfg["diffusion_model"]

    SavedSettingsGenerator.revision = revision

    def backend(name, **kwargs):
        calls.append((name, kwargs))
        return SavedSettingsGenerator()

    monkeypatch.setattr(human_study, "get_backend", backend)
    human_study.build_sheet(tmp_path, n=1)
    assert calls == [
        (
            "diffusers",
            {"model_id": cfg["diffusion_model"], "device": "cpu", "steps": 2, "revision": revision},
        )
    ]
    wrong = SavedSettingsGenerator()
    wrong.device = "mps"
    with pytest.raises(ValueError, match="supplied backend differs"):
        human_study.build_sheet(tmp_path, n=1, gen=wrong, out=tmp_path / "wrong-backend")


@pytest.mark.parametrize("destination_kind", ["empty", "file", "symlink", "broken_symlink"])
def test_existing_destination_is_refused_before_results_or_backend_work(
    tmp_path, monkeypatch, destination_kind
):
    destination = tmp_path / "existing"
    if destination_kind == "empty":
        destination.mkdir()
    elif destination_kind == "file":
        destination.write_text("preserved")
    else:
        target = tmp_path / "target"
        if destination_kind == "symlink":
            target.mkdir()
        destination.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        human_study, "get_backend", lambda *a, **kw: pytest.fail("backend initialized")
    )
    with pytest.raises(FileExistsError, match="Choose a new directory"):
        human_study.build_sheet(tmp_path / "no-results", out=destination)
    if destination_kind == "file":
        assert destination.read_text() == "preserved"


def test_rebuilding_does_not_overwrite_images_key_or_completed_ratings(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps(_saved_original_payload(tmp_path)))
    study = human_study.build_sheet(tmp_path, n=1)
    template = _read_csv(study / "ratings_template.csv")
    template[0]["emotion"] = "fear"
    human_study._write_csv(
        study / "rated.csv", [*human_study._IDENTITY_FIELDS, "emotion"], template
    )
    before = {path: path.read_bytes() for path in study.rglob("*") if path.is_file()}
    with pytest.raises(FileExistsError):
        human_study.build_sheet(tmp_path, n=1)
    assert {path: path.read_bytes() for path in study.rglob("*") if path.is_file()} == before


def test_fresh_studies_have_distinct_identities_bound_to_verified_pixels(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps(_saved_original_payload(tmp_path)))
    studies = [human_study.build_sheet(tmp_path, n=1, out=tmp_path / name) for name in ("a", "b")]
    keys = [_read_csv(study / "key.csv")[0] for study in studies]
    assert keys[0]["study_id"] != keys[1]["study_id"]
    assert keys[0]["id"] != keys[1]["id"]
    assert keys[0]["image_pixel_sha256"] == keys[1]["image_pixel_sha256"]
    for study, key in zip(studies, keys, strict=True):
        template = _read_csv(study / "ratings_template.csv")[0]
        assert all(template[field] == key[field] for field in human_study._IDENTITY_FIELDS)
        assert template["id"].startswith(template["study_id"] + "-")
        with Image.open(study / template["image"]) as image:
            assert run._image_digest(image) == template["image_pixel_sha256"]
    with pytest.raises(ValueError, match="image IDs absent from the key"):
        human_study.analyze(studies[0] / "ratings_template.csv", studies[1] / "key.csv")


def test_failed_build_does_not_publish_partial_study(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps(_saved_original_payload(tmp_path)))
    Image.new("RGB", (8, 8), "red").save(tmp_path / "original.png")
    destination = tmp_path / "study"
    with pytest.raises(ValueError, match="pixel digest differs"):
        human_study.build_sheet(tmp_path, n=1, out=destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".study-*"))


def test_destination_created_during_build_is_preserved(tmp_path, monkeypatch):
    (tmp_path / "results.json").write_text(json.dumps(_saved_original_payload(tmp_path)))
    destination = tmp_path / "study"
    original = human_study._write_csv

    def competing_directory(path, fields, rows):
        original(path, fields, rows)
        if not destination.exists():
            destination.mkdir()
            (destination / "ratings.csv").write_text("preserved")

    monkeypatch.setattr(human_study, "_write_csv", competing_directory)
    with pytest.raises(FileExistsError):
        human_study.build_sheet(tmp_path, n=1, out=destination)
    assert list(destination.iterdir()) == [destination / "ratings.csv"]
    assert (destination / "ratings.csv").read_text() == "preserved"
    assert not list(tmp_path.glob(".study-*"))


def test_publication_failure_removes_only_the_new_study(tmp_path, monkeypatch):
    (tmp_path / "results.json").write_text(json.dumps(_saved_original_payload(tmp_path)))
    previous = tmp_path / "previous-study"
    previous.mkdir()
    (previous / "rated.csv").write_text("preserved ratings")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    destination = tmp_path / "study"
    original = human_study.Path.replace
    moves = 0

    def failed_second_move(path, target):
        nonlocal moves
        moves += 1
        if moves == 2:
            raise OSError("publication failed")
        return original(path, target)

    monkeypatch.setattr(human_study.Path, "replace", failed_second_move)
    with pytest.raises(OSError, match="publication failed"):
        human_study.build_sheet(tmp_path, n=1, out=destination)
    assert moves == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".study-*"))
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("legacy_file", ["ratings", "key"])
def test_analyze_refuses_unbound_legacy_study_files(tmp_path, legacy_file):
    ratings, key = _write_study_files(
        tmp_path,
        [{**_identity(0), "emotion": "joy"}],
        [{**_identity(0), "intended": "joy", "probe": "joy"}],
    )
    if legacy_file == "ratings":
        human_study._write_csv(ratings, ["id", "emotion"], [{"id": 0, "emotion": "joy"}])
    else:
        human_study._write_csv(
            key, ["id", "intended", "probe"], [{"id": 0, "intended": "joy", "probe": "joy"}]
        )
    with pytest.raises(ValueError, match="Old unbound ratings cannot be migrated safely"):
        human_study.analyze(ratings, key)


@pytest.mark.parametrize("emotion", ["joy", ""])
def test_analyze_validates_hash_binding_even_for_unrated_rows(tmp_path, emotion):
    ratings, key = _write_study_files(
        tmp_path,
        [{**_identity(0), "image_pixel_sha256": "b" * 64, "emotion": emotion}],
        [{**_identity(0), "intended": "joy", "probe": "joy"}],
    )
    with pytest.raises(ValueError, match="do not match the study key"):
        human_study.analyze(ratings, key)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("study_id", "invalid", "Invalid study/image identity"),
        ("id", "0", "Invalid study/image identity"),
        ("image_pixel_sha256", "bad-hash", "Invalid image pixel digest"),
        ("image", "../original.png", "Invalid image path"),
    ],
)
def test_analyze_rejects_malformed_bindings(tmp_path, field, value, message):
    ratings, key = _write_study_files(
        tmp_path,
        [{**_identity(0), field: value, "emotion": "joy"}],
        [{**_identity(0), "intended": "joy", "probe": "joy"}],
    )
    with pytest.raises(ValueError, match=message):
        human_study.analyze(ratings, key)


@pytest.mark.parametrize("mixed_file", ["ratings", "key"])
def test_analyze_refuses_mixed_studies(tmp_path, mixed_file):
    identities = [_identity(0), _identity(1, study_id="2" * 32)]
    ratings = [{**_identity(0), "emotion": "joy"}]
    key = [{**_identity(0), "intended": "joy", "probe": "joy"}]
    if mixed_file == "ratings":
        ratings = [{**identity, "emotion": "joy"} for identity in identities]
    else:
        key = [{**identity, "intended": "joy", "probe": "joy"} for identity in identities]
    ratings_path, key_path = _write_study_files(tmp_path, ratings, key)
    with pytest.raises(ValueError, match="Multiple study IDs"):
        human_study.analyze(ratings_path, key_path)


@pytest.mark.parametrize("mutation", ["changed", "missing", "unreadable", "broken_directory"])
def test_analyze_verifies_available_study_images(tmp_path, mutation):
    (tmp_path / "results.json").write_text(json.dumps(_saved_original_payload(tmp_path)))
    study = human_study.build_sheet(tmp_path, n=1)
    identity = _read_csv(study / "key.csv")[0]
    image_path = study / identity["image"]
    if mutation == "changed":
        Image.new("RGB", (8, 8), "red").save(image_path)
    elif mutation == "unreadable":
        image_path.write_bytes(b"not an image")
    else:
        image_path.unlink()
        if mutation == "broken_directory":
            (study / "images").rmdir()
            (study / "images").symlink_to(study / "absent", target_is_directory=True)
    with pytest.raises(ValueError, match="pixel digest differs|image is missing|cannot be read"):
        human_study.analyze(study / "ratings_template.csv", study / "key.csv")


def test_analyze_rejects_unknown_key_labels(tmp_path):
    ratings, key = _write_study_files(
        tmp_path,
        [{**_identity(0), "emotion": "joy"}],
        [{**_identity(0), "intended": "positive", "probe": "joy"}],
    )
    with pytest.raises(ValueError, match="unknown emotion label"):
        human_study.analyze(ratings, key)
