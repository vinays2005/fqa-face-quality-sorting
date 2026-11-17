import json, os, pytest
from fqa.reclassify import safe_slug, safe_copy


@pytest.mark.parametrize("bad", ["../evil", "..", "a/b", "", ".hidden", "x\\u0000y", "a..b"])
def test_safe_slug_rejects(bad):
    if bad == "":
        assert safe_slug(bad, "fallback") == "fallback"
    else:
        with pytest.raises(ValueError):
            safe_slug(bad)


def test_safe_copy_rejects_outside_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    src = tmp_path / "outside.jpg"
    src.write_bytes(b"x")
    with pytest.raises(ValueError):
        safe_copy(str(src), str(tmp_path / "out"), str(root), str(tmp_path / "out"))


def test_safe_copy_rejects_symlink(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    real = root / "real.jpg"
    real.write_bytes(b"x")
    link = root / "link.jpg"
    if hasattr(os, "symlink"):
        os.symlink(real, link)
        out = tmp_path / "out"
        out.mkdir()
        with pytest.raises(ValueError):
            safe_copy(str(link), str(out), str(root), str(out))


def test_safe_copy_dedups_collisions(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    a = root / "x.jpg"
    a.write_bytes(b"a")
    b = root / "sub" / "x.jpg"
    b.parent.mkdir()
    b.write_bytes(b"b")
    safe_copy(str(a), str(out), str(root), str(out))
    p2 = safe_copy(str(b), str(out), str(root), str(out))
    assert os.path.basename(p2) != "x.jpg"