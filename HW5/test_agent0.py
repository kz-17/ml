#!/usr/bin/env python3
# test_agent0.py - Security tests for agent0

import os
import sys
import tempfile

import agent0


def test_resolve_path():
    """Test path resolution with expanduser and expandvars."""
    # Relative path -> should join with WORKSPACE
    result = agent0.resolve_path("test.txt")
    expected = os.path.join(agent0.WORKSPACE, "test.txt")
    assert os.path.abspath(result) == os.path.abspath(expected), \
        f"resolve_path('test.txt') = {result}, expected {expected}"

    # Absolute path -> should stay as-is
    abs_path = os.path.abspath(os.path.join(agent0.WORKSPACE, "..", "outside.txt"))
    result = agent0.resolve_path(abs_path)
    assert result == abs_path, f"resolve_path absolute returned {result}"

    # Path with ~ -> should expand home
    home = os.path.expanduser("~")
    result = agent0.resolve_path("~/test.txt")
    assert result.startswith(home), \
        f"resolve_path('~/test.txt') = {result}, expected to start with {home}"

    # Path with env var -> should expand
    result = agent0.resolve_path("%USERPROFILE%/test.txt" if os.name == 'nt'
                                 else "$HOME/test.txt")
    assert os.path.isabs(result), f"expandvars path should be absolute, got {result}"

    print("  resolve_path .................................. PASS")


def test_is_path_safe():
    """Test safety boundary enforcement."""
    # Inside workspace -> safe
    inside = os.path.join(agent0.WORKSPACE, "some_file.txt")
    assert agent0.is_path_safe(inside), "Path inside workspace should be safe"

    # Outside workspace -> unsafe
    outside = os.path.abspath(os.path.join(agent0.WORKSPACE, "..", "outside.txt"))
    assert not agent0.is_path_safe(outside), "Path outside workspace should be unsafe"

    # Workspace itself -> safe
    assert agent0.is_path_safe(agent0.WORKSPACE), "Workspace itself should be safe"

    # Subdir inside workspace -> safe
    subdir = os.path.join(agent0.WORKSPACE, "subdir", "file.txt")
    assert agent0.is_path_safe(subdir), "Subdirectory should be safe"

    # Sibling directory -> unsafe
    sibling = os.path.abspath(os.path.join(agent0.WORKSPACE, os.pardir, "sibling"))
    assert not agent0.is_path_safe(sibling), "Sibling directory should be unsafe"

    print("  is_path_safe ................................. PASS")


def test_extract_referenced_paths():
    """Test path extraction from shell commands."""
    # Simple unsafe path
    paths = agent0.extract_referenced_paths("cat ../outside.txt")
    assert any(".." in p or "outside.txt" in p for p in paths), \
        f"Should find '../outside.txt', got {paths}"

    # Quoted path
    paths = agent0.extract_referenced_paths('type "C:\\Users\\test.txt"')
    assert any("test.txt" in p or "C:" in p for p in paths), \
        f"Should find quoted path, got {paths}"

    # No path (pure command)
    paths = agent0.extract_referenced_paths("echo hello world")
    assert len(paths) == 0, \
        f"Should not extract from 'echo hello world', got {paths}"

    # Windows drive letter
    paths = agent0.extract_referenced_paths("type D:\\data\\file.csv")
    assert any("D:" in p for p in paths), \
        f"Should find Windows path, got {paths}"

    # Relative with backslash (Windows)
    paths = agent0.extract_referenced_paths("type ..\\secret.txt")
    assert any(".." in p or "secret" in p for p in paths), \
        f"Should find '..\\secret.txt', got {paths}"

    # Empty command
    paths = agent0.extract_referenced_paths("")
    assert paths == [], "Empty command should return empty list"

    # None command
    paths = agent0.extract_referenced_paths(None)
    assert paths == [], "None should return empty list"

    print("  extract_referenced_paths .................... PASS")


def test_check_shell_safety():
    """Test shell command safety analysis."""
    # Safe commands (inside workspace or no file access)
    safe_cmds = [
        "echo hello",
        "dir",
        f"type {os.path.join(agent0.WORKSPACE, 'test.txt')}",
    ]
    for cmd in safe_cmds:
        unsafe = agent0.check_shell_safety(cmd)
        if unsafe:
            print(f"    ⚠  False positive for: {cmd} -> {unsafe}")

    # Unsafe command (parent dir access)
    sep = "\\" if os.name == "nt" else "/"
    cmd = f"type ..{sep}outside.txt"
    unsafe = agent0.check_shell_safety(cmd)
    assert len(unsafe) > 0, \
        f"Should detect '{cmd}' as unsafe"

    # Unsafe absolute path
    cmd = "/etc/passwd" if os.name != "nt" else "type C:\\Windows\\system.ini"
    # Some absolute paths might not resolve on this system, but we test detection
    paths = agent0.extract_referenced_paths(cmd)
    assert len(paths) > 0, f"Should extract paths from '{cmd}'"

    print("  check_shell_safety .......................... PASS")


def test_file_operations():
    """Test read/write with sandbox enforcement."""
    test_file = "_test_security_temp.txt"
    test_content = "Hello Security!"

    # ─── Write inside workspace (should succeed) ───
    agent0.AUTO_AUTHORIZE = True
    result = agent0.handle_write_file(test_file, test_content)
    assert "Successfully" in result, f"Write inside workspace failed: {result}"

    # Verify file was created
    full_path = os.path.join(agent0.WORKSPACE, test_file)
    assert os.path.exists(full_path), f"File was not created at {full_path}"

    # ─── Read inside workspace (should succeed) ───
    content = agent0.handle_read_file(test_file)
    assert content == test_content, f"Read inside workspace failed: got '{content}'"

    # Clean up internal file
    os.remove(full_path)
    assert not os.path.exists(full_path), "Cleanup failed"

    # ─── Write outside workspace (should be denied) ───
    agent0.AUTO_AUTHORIZE = False
    agent0.TEST_DENY_EXTERNAL = True
    outside_path = os.path.abspath(os.path.join(agent0.WORKSPACE, os.pardir, "_hack_test.txt"))
    result = agent0.handle_write_file(outside_path, "evil content")
    assert "Permission denied" in result, \
        f"Write outside should be denied, got: {result}"

    # Verify no file was created externally
    assert not os.path.exists(outside_path), \
        "File was created outside workspace despite denial!"

    # ─── Read outside workspace (should be denied) ───
    result = agent0.handle_read_file(outside_path)
    assert "Permission denied" in result, \
        f"Read outside should be denied, got: {result}"

    # ─── Reset flags ───
    agent0.TEST_DENY_EXTERNAL = False
    agent0.AUTO_AUTHORIZE = False

    print("  file_operations ............................. PASS")


def test_edge_cases():
    """Test edge cases for path safety."""
    # Path with symlink-like traversal
    tricky = os.path.join(agent0.WORKSPACE, "subdir", os.pardir, "escape.txt")
    resolved = agent0.resolve_path(tricky)
    expected = os.path.join(agent0.WORKSPACE, "escape.txt")
    assert agent0.is_path_safe(resolved), \
        f"Path with pardir inside WS should be safe, got {resolved}"

    # Double-dot escape
    escaped = os.path.abspath(os.path.join(agent0.WORKSPACE, os.pardir, os.pardir, "etc"))
    assert not agent0.is_path_safe(escaped), \
        "Double parent escape should be unsafe"

    # Same path as workspace
    assert agent0.is_path_safe(agent0.WORKSPACE), \
        "Workspace itself should be safe"

    # Empty-ish path (relative dot)
    dot_path = agent0.resolve_path(".")
    assert agent0.is_path_safe(dot_path), \
        "Dot path should resolve to workspace and be safe"

    print("  edge_cases .................................. PASS")


def test_authorize_flag_toggle():
    """Test that authorization flags work correctly."""
    # Default state
    assert not agent0.AUTO_AUTHORIZE, "AUTO_AUTHORIZE should default to False"
    assert not agent0.TEST_DENY_EXTERNAL, "TEST_DENY_EXTERNAL should default to False"

    # Toggle AUTO_AUTHORIZE
    agent0.AUTO_AUTHORIZE = True
    assert agent0.authorize("anything") == True, "AUTO_AUTHORIZE should always return True"
    agent0.AUTO_AUTHORIZE = False

    # Toggle TEST_DENY_EXTERNAL
    agent0.TEST_DENY_EXTERNAL = True
    assert agent0.authorize("外部檔案存取") == False, "TEST_DENY should deny external"
    # Non-external message with TEST_DENY should fall through to input (skip in test)
    agent0.TEST_DENY_EXTERNAL = False

    print("  authorize_flag_toggle ....................... PASS")


def main():
    print(f"\n{'='*55}")
    print("  Agent0 安全功能測試")
    print(f"{'='*55}")
    print(f"  工作區：{agent0.WORKSPACE}")
    print(f"  平台：{sys.platform}")
    print()

    tests = [
        test_resolve_path,
        test_is_path_safe,
        test_extract_referenced_paths,
        test_check_shell_safety,
        test_file_operations,
        test_edge_cases,
        test_authorize_flag_toggle,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  {test.__name__.ljust(35)} FAIL")
            print(f"    {e}")
            failed += 1

    print(f"\n{'='*55}")
    print(f"  結果：{passed} 通過, {failed} 失敗 / {len(tests)} 項")
    print(f"{'='*55}\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
