"""End-to-end integration tests for Minecraft texture pipeline (Phase 1)."""

import io
import tempfile

from PIL import Image

from twin.core.config import Settings
from twin.models.clip_model import load as load_model
from twin.services.index_service import index_single
from twin.services.indexer import indexer
from twin.services.search import search

load_model()


def _create_mock_64x32_skin() -> Image.Image:
    """Create a 64x32 image with arm/torso pixels populated to classify as SKIN_SINGLE."""
    img = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    # Fill head & torso
    for x in range(0, 32):
        for y in range(0, 32):
            img.putpixel((x, y), (200, 150, 100, 255))
    # Right arm (x: 44..48, y: 20..32)
    for x in range(44, 48):
        for y in range(20, 32):
            img.putpixel((x, y), (180, 130, 90, 255))
    return img


def _create_mock_64x64_skin() -> Image.Image:
    """Create a 64x64 modern skin."""
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    for x in range(0, 64):
        for y in range(0, 64):
            img.putpixel((x, y), (100, 200, 150, 255))
    return img


def _create_mock_cape() -> Image.Image:
    """Create a 64x32 cape with cape region populated."""
    img = Image.new("RGBA", (64, 32), (0, 0, 0, 0))
    for x in range(1, 22):
        for y in range(1, 17):
            img.putpixel((x, y), (255, 0, 0, 255))
    return img


def _to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_index_single_routing_and_upgrade(monkeypatch):
    """Test that index_single inspects textures, upgrades 64x32, and routes properly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(index_path=tmpdir, images_dir=tmpdir, faiss_index_type="flat")
        monkeypatch.setattr("twin.services.indexer.settings", s)
        monkeypatch.setattr("twin.services.index_service.settings", s)
        monkeypatch.setattr("twin.services.search.settings", s)

        indexer.load()

        # 1. Index 64x32 old skin
        skin32 = _create_mock_64x32_skin()
        res32 = index_single(skin32, "old_skin.png", _to_bytes(skin32))
        assert res32["status"] == "indexed"
        assert res32["datasource"] == "skin"
        meta32 = indexer.get_metadata(res32["id"], datasource="skin")
        assert meta32["upgraded"] is True
        assert meta32["texture_type"] == "skin_single"

        # 2. Index 64x64 modern skin
        skin64 = _create_mock_64x64_skin()
        res64 = index_single(skin64, "new_skin.png", _to_bytes(skin64))
        assert res64["status"] == "indexed"
        assert res64["datasource"] == "skin"
        meta64 = indexer.get_metadata(res64["id"], datasource="skin")
        assert meta64["upgraded"] is False
        assert meta64["texture_type"] == "skin_double"

        # 3. Index cape
        cape = _create_mock_cape()
        resc = index_single(cape, "cape.png", _to_bytes(cape))
        assert resc["status"] == "indexed"
        assert resc["datasource"] == "cape"
        metac = indexer.get_metadata(resc["id"], datasource="cape")
        assert metac["upgraded"] is False
        assert metac["texture_type"] == "cape"

        # Verify separation in counts
        assert indexer.get_count("skin") == 2
        assert indexer.get_count("cape") == 1


def test_search_auto_routing_and_cross_version(monkeypatch):
    """Test that search auto-routes queries and aligns 64x32 queries."""
    with tempfile.TemporaryDirectory() as tmpdir:
        s = Settings(index_path=tmpdir, images_dir=tmpdir, faiss_index_type="flat")
        monkeypatch.setattr("twin.services.indexer.settings", s)
        monkeypatch.setattr("twin.services.index_service.settings", s)
        monkeypatch.setattr("twin.services.search.settings", s)

        indexer.load()

        # Index skins
        skin32 = _create_mock_64x32_skin()
        index_single(skin32, "target_skin.png", _to_bytes(skin32))

        cape = _create_mock_cape()
        index_single(cape, "target_cape.png", _to_bytes(cape))

        # Search with a 64x32 skin query
        res_skin = search(skin32, top_k=5)
        assert res_skin["datasource"] == "skin"
        assert res_skin["count"] > 0
        assert res_skin["results"][0]["filename"] == "target_skin.png"

        # Search with a cape query
        res_cape = search(cape, top_k=5)
        assert res_cape["datasource"] == "cape"
        assert res_cape["count"] > 0
        assert res_cape["results"][0]["filename"] == "target_cape.png"
