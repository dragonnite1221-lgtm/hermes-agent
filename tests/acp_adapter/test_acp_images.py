import base64

import pytest
from acp.schema import (
    BlobResourceContents,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
)

from acp_adapter.server import HermesACPAgent, _content_blocks_to_openai_user_content


def test_acp_image_blocks_convert_to_openai_multimodal_content():
    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="What is in this image?"),
        ImageContentBlock(type="image", data="aGVsbG8=", mimeType="image/png"),
    ])

    assert content == [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,aGVsbG8="},
        },
    ]


def test_audio_only_prompt_becomes_a_text_placeholder_not_empty():
    """An AudioContentBlock has no real conversion path (no canonical
    "build an input_audio part" helper in this codebase, and provider
    audio-input support/format varies), but dropping it silently would
    turn an audio-only prompt into a completely empty one that never
    reaches the agent at all. It must at least surface as a text
    placeholder instead of vanishing without a trace.
    """
    from acp.schema import AudioContentBlock

    content = _content_blocks_to_openai_user_content([
        AudioContentBlock(type="audio", data="aGVsbG8=", mimeType="audio/wav"),
    ])

    assert isinstance(content, str)
    assert "audio/wav" in content
    assert content.strip()


def test_snapshot_resource_link_now_handles_oversized_and_mimeless_images(tmp_path):
    """snapshot_resource_link_now must mirror _resource_link_to_parts'
    two image-specific behaviors instead of treating every resource_link
    as generic bytes:

    1. An oversized image is never read (let alone truncated) -- a
       truncated blob would replay as a corrupt, non-decodable
       image_url instead of a clear "too large" notice.
    2. A resource_link with no explicit mimeType is still recognized as
       an image via its file suffix, so it embeds as an image blob
       instead of "binary embedded file omitted".
    """
    from acp_adapter.content import _MAX_ACP_RESOURCE_BYTES, snapshot_resource_link_now

    oversized = tmp_path / "huge.png"
    oversized.write_bytes(b"\x89PNG" + b"\x00" * (_MAX_ACP_RESOURCE_BYTES + 1))
    oversized_block = ResourceContentBlock(
        type="resource_link", name="huge.png", uri=oversized.as_uri(), mimeType="image/png"
    )
    snapshotted = snapshot_resource_link_now(oversized_block)
    assert isinstance(snapshotted, EmbeddedResourceContentBlock)
    assert isinstance(snapshotted.resource, TextResourceContents)
    assert "too large" in snapshotted.resource.text.lower()

    small_png_bytes = base64.b64decode("aGVsbG8=")
    mimeless = tmp_path / "photo.png"
    mimeless.write_bytes(small_png_bytes)
    mimeless_block = ResourceContentBlock(type="resource_link", name="photo.png", uri=mimeless.as_uri())
    snapshotted_mimeless = snapshot_resource_link_now(mimeless_block)
    assert isinstance(snapshotted_mimeless, EmbeddedResourceContentBlock)
    assert isinstance(snapshotted_mimeless.resource, BlobResourceContents)
    assert snapshotted_mimeless.resource.mime_type == "image/png"
    assert base64.b64decode(snapshotted_mimeless.resource.blob) == small_png_bytes


def test_snapshot_resource_link_now_preserves_name_and_title_through_replay(tmp_path):
    """A resource_link's name/title must survive snapshot_resource_link_now and still show up
    when the snapshot is later rendered to OpenAI content -- not fall back to the URI's bare
    basename, which is often an opaque temp/cache path an editor uses on disk while the real
    resource identity lives in ``name``/``title``.
    """
    from acp_adapter.content import _content_blocks_to_openai_user_content, snapshot_resource_link_now

    attached = tmp_path / "tmp_cache_ab12.md"
    attached.write_text("hello world", encoding="utf-8")
    block = ResourceContentBlock(
        type="resource_link", name="tmp_cache_ab12.md", title="design-notes.md", uri=attached.as_uri(),
        mimeType="text/markdown",
    )

    snapshotted = snapshot_resource_link_now(block)
    content = _content_blocks_to_openai_user_content([snapshotted])

    assert isinstance(content, str)
    assert "design-notes.md" in content
    assert "tmp_cache_ab12.md" in content


def test_snapshot_resource_link_now_bounds_and_flags_oversized_text(tmp_path):
    """A queued non-image resource bigger than the cap must be bounded to
    _MAX_ACP_RESOURCE_BYTES (never the whole file pulled into memory) AND the replayed
    content must say the resource was truncated -- otherwise a queued turn silently treats a
    partial file read as the complete attachment.
    """
    from acp_adapter.content import (
        _MAX_ACP_RESOURCE_BYTES,
        _content_blocks_to_openai_user_content,
        snapshot_resource_link_now,
    )

    real_size = _MAX_ACP_RESOURCE_BYTES + 1000
    oversized = tmp_path / "huge.txt"
    oversized.write_text("A" * real_size, encoding="utf-8")
    block = ResourceContentBlock(type="resource_link", name="huge.txt", uri=oversized.as_uri())

    snapshotted = snapshot_resource_link_now(block)
    assert isinstance(snapshotted, EmbeddedResourceContentBlock)
    assert isinstance(snapshotted.resource, TextResourceContents)
    # Bounded, not the whole file: the stored text must stay near the cap, nowhere close to
    # the real file size.
    assert len(snapshotted.resource.text) <= _MAX_ACP_RESOURCE_BYTES + 200

    content = _content_blocks_to_openai_user_content([snapshotted])
    assert isinstance(content, str)
    assert "truncated" in content.lower()
    assert str(real_size) in content


def test_text_only_acp_blocks_stay_string_for_legacy_prompt_path():
    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="/help"),
    ])

    assert content == "/help"


def test_acp_resource_link_file_is_inlined_as_text(tmp_path):
    attached = tmp_path / "notes.md"
    attached.write_text("# Notes\n\nAttached file body", encoding="utf-8")

    content = _content_blocks_to_openai_user_content([
        TextContentBlock(type="text", text="Please read this file"),
        ResourceContentBlock(
            type="resource_link",
            name="notes.md",
            title="Project notes",
            uri=attached.as_uri(),
            mimeType="text/markdown",
        ),
    ])

    assert content == (
        "Please read this file\n"
        "[Attached file: Project notes (notes.md)]\n"
        f"URI: {attached.as_uri()}\n\n"
        "# Notes\n\nAttached file body"
    )




@pytest.mark.asyncio
async def test_initialize_advertises_image_prompt_capability():
    response = await HermesACPAgent().initialize()

    assert response.agent_capabilities is not None
    assert response.agent_capabilities.prompt_capabilities is not None
    assert response.agent_capabilities.prompt_capabilities.image is True


# 1x1 transparent PNG — smallest valid image payload for inlining tests.
_ONE_PX_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6300010000000500010d0a2db40000000049454e44ae426082"
)






