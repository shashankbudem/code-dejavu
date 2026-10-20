"""Smoke test for the commit check: `python test_dejavu.py` (loads EmbeddingGemma 2, ~30 s)."""
import os
import subprocess
import tempfile

FORMAT_BYTES = """export function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${Math.ceil(bytes / 1024)} KB`;
  }
  return `${Math.round(bytes / 1024 / 1024)} MB`;
}
"""
PARSE_CONFIG = """export function parseConfig(raw: string) {
  const parsed = JSON.parse(raw);
  if (!parsed.name) {
    throw new Error("config needs a name");
  }
  return { name: parsed.name, port: parsed.port ?? 8080 };
}
"""
# Copy-pasted under a new name: must be flagged. (Renaming variables too scores ~0.95, right at the cutoff,
# and a from-scratch rewrite lands in the borderline band, so either would make this test fragile.)
NEAR_COPY = FORMAT_BYTES.replace("formatBytes", "sizeLabel")
UNRELATED = """export function slugify(title: string): string {
  return title
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}
"""


def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True, capture_output=True)


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write(text)


with tempfile.TemporaryDirectory() as tmp:
    tmp = os.path.realpath(tmp)
    os.environ["DEJAVU_HOME"] = f"{tmp}/home"
    import dejavu  # reads DEJAVU_HOME on import

    lib, app = f"{tmp}/lib", f"{tmp}/app"
    write(f"{lib}/src/size.ts", FORMAT_BYTES)
    write(f"{app}/src/config.ts", PARSE_CONFIG)
    git(app, "init", "-q")
    git(app, "add", ".")
    git(app, "commit", "-qm", "init")
    dejavu.build_index([lib, app])

    write(f"{app}/src/ui.ts", NEAR_COPY + "\n" + UNRELATED)
    write(f"{app}/src/config.ts", PARSE_CONFIG.replace("8080", "3000"))  # edit an already-indexed function
    git(app, "add", ".")
    found = {f["name"]: f for f in dejavu.check(app)}
    dejavu.report(found.values())

    assert found.get("sizeLabel", {}).get("level") == "warn", "near-copy of formatBytes was not flagged"
    assert found["sizeLabel"]["match"]["name"] == "formatBytes"
    assert "slugify" not in found, "unrelated function was flagged"
    assert "parseConfig" not in found, "an edited function was flagged against its own old version"
    print("ok")
