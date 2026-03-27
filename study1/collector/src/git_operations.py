"""Git operations module for cloning and managing repositories.

This module provides utilities for:
- Verifying git installation
- Cloning repositories (HTTPS and SSH)
- Checking out branches
- Validating git repositories
"""
import base64
import os
import subprocess
import shutil
import re
import sys
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse, urlunparse
import requests


def parse_target(target: str) -> dict:
    """Parse a CLI target URL into mode, git_type, org, and repo_url.

    Args:
        target: An HTTPS org/repo URL or an SSH repo URL.

    Returns:
        dict with keys:
          - mode: "single-repo" or "batch"
          - git_type: "github", "gitlab", "azure", or "bitbucket"
          - org: organisation / group path extracted from the URL
          - repo_url: the full clone URL (single-repo) or None (batch)

    Raises:
        ValueError: If the domain doesn't match a known git provider.
    """
    target = target.strip()

    # SSH URL: git@github.com:org/repo.git
    ssh_match = re.match(r'^git@([^:]+):(.+)$', target)
    if ssh_match:
        domain = ssh_match.group(1).lower()
        path = ssh_match.group(2).rstrip('/')
        git_type = _domain_to_git_type(domain)
        # SSH is always single-repo (always has .git)
        org = '/'.join(path.split('/')[:-1])
        if not org:
            raise ValueError(f"Cannot extract organisation from SSH URL: {target}")
        return {"mode": "single-repo", "git_type": git_type, "org": org, "repo_url": target}

    # HTTPS URL
    parsed = urlparse(target)
    if not parsed.hostname:
        raise ValueError(f"Invalid target URL: {target}")
    domain = parsed.hostname.lower()
    git_type = _domain_to_git_type(domain)

    path = parsed.path.strip('/')

    # Azure DevOps special handling: dev.azure.com/{org}/{project}/_git/{repo}
    if git_type == 'azure' and '/_git/' in parsed.path:
        parts = path.split('/_git/')
        org = parts[0]  # org/project
        return {"mode": "single-repo", "git_type": git_type, "org": org, "repo_url": target}

    if path.endswith('.git'):
        # Single-repo mode
        path_no_git = path[:-4]
        segments = path_no_git.split('/')
        org = '/'.join(segments[:-1])
        if not org:
            raise ValueError(f"Cannot extract organisation from URL: {target}")
        return {"mode": "single-repo", "git_type": git_type, "org": org, "repo_url": target}
    else:
        # Batch mode
        if not path:
            raise ValueError(f"Cannot extract organisation from URL: {target}")
        return {"mode": "batch", "git_type": git_type, "org": path, "repo_url": None}


def _domain_to_git_type(domain: str) -> str:
    """Map a hostname to a git provider type.

    Raises:
        ValueError: If the domain doesn't match any known provider.
    """
    if domain == 'github.com':
        return 'github'
    if domain == 'gitlab.com' or 'gitlab' in domain:
        return 'gitlab'
    if 'dev.azure.com' in domain or 'visualstudio.com' in domain:
        return 'azure'
    if domain == 'bitbucket.org' or 'bitbucket' in domain:
        return 'bitbucket'
    raise ValueError(
        f"Unknown git provider for domain '{domain}'. "
        f"Supported: github.com, gitlab.com, dev.azure.com, bitbucket.org"
    )


def verify_git_installed() -> bool:
    """Check if git is available on the system.

    Returns:
        bool: True if git is installed, False otherwise
    """
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def is_git_repository(path: str) -> bool:
    """Check if a directory is a valid git repository.

    Args:
        path: Path to the directory to check

    Returns:
        bool: True if valid git repository, False otherwise
    """
    git_dir = Path(path) / ".git"
    if git_dir.exists():
        return True

    # Also try running git command to be sure
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def get_current_branch(repo_path: str) -> str:
    """Get the currently checked out branch name.

    Args:
        repo_path: Path to the git repository

    Returns:
        str: Name of the current branch

    Raises:
        Exception: If unable to get branch name
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to get current branch: {e.stderr}")


def checkout_branch(repo_path: str, branch: str) -> bool:
    """Checkout a specific branch.

    Args:
        repo_path: Path to the git repository
        branch: Name of the branch to checkout

    Returns:
        bool: True if successful

    Raises:
        Exception: If branch doesn't exist or checkout fails
    """
    try:
        result = subprocess.run(
            ["git", "checkout", branch],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to checkout branch '{branch}': {e.stderr}")


def pull_latest(repo_path: str, branch: str, timeout: int = 300) -> bool:
    """Pull the latest changes from the remote for the given branch.

    Args:
        repo_path: Path to the git repository
        branch: Name of the branch to pull
        timeout: Timeout in seconds for the git pull operation

    Returns:
        bool: True if successful

    Raises:
        Exception: If pull fails
    """
    # Use idle timeout — watch .git dir for incoming objects
    git_dir = str(Path(repo_path) / ".git")
    try:
        _run_with_idle_timeout(
            ["git", "pull", "origin", branch],
            idle_timeout=timeout,
            watch_path=git_dir,
            cwd=repo_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        stderr = e.stderr or ""
        raise Exception(f"Failed to pull latest changes for '{branch}': {stderr}")
    except subprocess.TimeoutExpired:
        raise Exception(f"Pull operation timed out for branch '{branch}' (no progress for {timeout}s)")


def find_commit_at_date(repo_path: str, branch: str, end_date: str) -> Optional[str]:
    """Find the last commit on *branch* on or before *end_date*.

    Args:
        repo_path: Path to the git repository.
        branch: Branch to search.
        end_date: ISO 8601 date string (e.g. "2025-03-31").

    Returns:
        Commit SHA as a string, or ``None`` if no commit exists before
        *end_date*.
    """
    # --before is exclusive in git, so push the boundary to end-of-day
    cutoff = f"{end_date}T23:59:59"
    try:
        result = subprocess.run(
            ["git", "log", branch, f"--before={cutoff}", "--format=%H", "-n", "1"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        sha = result.stdout.strip()
        return sha if sha else None
    except subprocess.CalledProcessError:
        return None


def reset_to_commit(repo_path: str, commit_sha: str) -> None:
    """Hard-reset the current branch to *commit_sha*.

    This moves the branch pointer back to the given commit so the working
    tree reflects the repository state at that point in time.

    Args:
        repo_path: Path to the git repository.
        commit_sha: Full or abbreviated commit hash.

    Raises:
        Exception: If the reset fails.
    """
    try:
        subprocess.run(
            ["git", "reset", "--hard", commit_sha],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to reset to {commit_sha}: {e.stderr}")


def _extract_repo_name(repo_url: str) -> str:
    """Extract repository name from URL.

    Args:
        repo_url: Git repository URL

    Returns:
        str: Repository name
    """
    # Handle HTTPS URLs: https://github.com/user/repo.git
    # Handle SSH URLs: git@github.com:user/repo.git

    # Remove .git suffix if present
    url = repo_url.rstrip('/')
    if url.endswith('.git'):
        url = url[:-4]

    # Extract last part of path
    if '/' in url:
        repo_name = url.split('/')[-1]
    elif ':' in url:  # SSH format
        repo_name = url.split(':')[-1].split('/')[-1]
    else:
        repo_name = url

    return repo_name


def extract_qualified_repo_name(repo_url: str) -> str:
    """Extract owner__repo name from URL to avoid collisions.

    Returns 'owner__repo' for URLs like https://github.com/owner/repo.git.
    Falls back to bare repo name if owner cannot be determined.
    """
    url = repo_url.strip().rstrip('/')
    if url.endswith('.git'):
        url = url[:-4]

    # SSH format: git@github.com:owner/repo
    ssh_match = re.match(r'^git@[^:]+:(.+)$', url)
    if ssh_match:
        parts = ssh_match.group(1).split('/')
        if len(parts) >= 2:
            return f"{parts[-2]}__{parts[-1]}"
        return parts[-1]

    # HTTPS format: https://github.com/owner/repo
    parts = url.split('/')
    if len(parts) >= 2:
        return f"{parts[-2]}__{parts[-1]}"
    return parts[-1]

def _build_authenticated_url(repo_url: str, token: Optional[str]) -> str:
    """Embed a token into an HTTPS repo URL for git operations.

    Returns the original URL unchanged when *token* is None or the URL
    is not HTTP(S).
    """
    if not token:
        return repo_url

    if not repo_url.startswith(('http://', 'https://')):
        raise ValueError("Personal Access Tokens can only be used with HTTP/HTTPS URLs, not SSH.")

    parsed = urlparse(repo_url)
    domain = parsed.hostname.lower()

    if 'dev.azure.com' in domain or 'visualstudio.com' in domain:
        auth_prefix = f"AzDevOps:{token}"
    elif 'gitlab.com' in domain or 'gitlab' in domain:
        auth_prefix = f"oauth2:{token}"
    elif 'bitbucket.org' in domain or 'bitbucket' in domain:
        auth_prefix = f"x-token-auth:{token}"
    else:
        auth_prefix = f"{token}"

    clean_address = parsed.hostname
    if parsed.port:
        clean_address = f"{clean_address}:{parsed.port}"

    new_netloc = f"{auth_prefix}@{clean_address}"
    return urlunparse((
        parsed.scheme,
        new_netloc,
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))


def detect_default_branch(repo_url: str, token: Optional[str] = None, timeout: int = 30) -> Optional[str]:
    """Detect the default branch of a remote repository via ``git ls-remote``.

    Returns the branch name (e.g. ``"main"``) or *None* if detection fails.
    """
    url = _build_authenticated_url(repo_url, token)
    try:
        result = subprocess.run(
            ["git", "ls-remote", "--symref", url, "HEAD"],
            capture_output=True, text=True, check=True, timeout=timeout,
        )
        for line in result.stdout.splitlines():
            if line.startswith("ref:") and "HEAD" in line:
                ref = line.split()[1]
                if ref.startswith("refs/heads/"):
                    return ref.replace("refs/heads/", "")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return None


def _run_with_idle_timeout(cmd, idle_timeout, watch_path=None, **kwargs):
    """Run a subprocess with an idle-based timeout.

    Instead of a fixed total timeout, the process is only killed if no
    progress is detected for *idle_timeout* seconds.  Progress is measured
    by checking whether *watch_path* (directory or file) has changed in
    size since the last check.  If *watch_path* is ``None`` the function
    falls back to a regular total timeout.

    Returns a ``subprocess.CompletedProcess``-like object.

    Raises:
        subprocess.TimeoutExpired: if idle timeout is exceeded.
        subprocess.CalledProcessError: if the process exits non-zero.
    """
    if watch_path is None:
        return subprocess.run(cmd, timeout=idle_timeout, **kwargs)

    check = kwargs.pop("check", False)
    poll_interval = 5  # seconds between progress checks
    proc = subprocess.Popen(cmd, **kwargs)
    last_size = -1
    last_progress_time = time.monotonic()

    try:
        while True:
            try:
                proc.wait(timeout=poll_interval)
                # Process finished
                break
            except subprocess.TimeoutExpired:
                pass  # still running — check progress

            # Measure directory size (fast: just sum immediate children sizes)
            current_size = 0
            watch = Path(watch_path)
            if watch.exists():
                try:
                    if watch.is_dir():
                        for entry in os.scandir(str(watch)):
                            try:
                                current_size += entry.stat(follow_symlinks=False).st_size
                            except OSError:
                                pass
                    else:
                        current_size = watch.stat().st_size
                except OSError:
                    pass

            if current_size != last_size:
                last_size = current_size
                last_progress_time = time.monotonic()

            elapsed_idle = time.monotonic() - last_progress_time
            if elapsed_idle > idle_timeout:
                proc.kill()
                proc.wait()
                raise subprocess.TimeoutExpired(cmd, idle_timeout)

        # Check return code
        stdout = proc.stdout.read() if proc.stdout else None
        stderr = proc.stderr.read() if proc.stderr else None
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=stdout, stderr=stderr,
            )
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    except BaseException:
        proc.kill()
        proc.wait()
        raise


def clone_repository(repo_url: str, clone_dir: str, branch: str = "main", token: Optional[str] = None, timeout: int = 300) -> str:
    """Clone a git repository and checkout specified branch with provider-aware auth."""

    # Check if git is installed (assuming helper exists)
    if not verify_git_installed():
        raise Exception("Git is not installed on this system")

    # Validate URL format
    if not re.match(r'^(https?://|git@|ssh://)', repo_url) and '://' not in repo_url:
        raise Exception(f"Invalid URL format: {repo_url}")

    authenticated_url = _build_authenticated_url(repo_url, token)

    # Create clone_dir
    clone_path = Path(clone_dir)
    try:
        clone_path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        raise PermissionError(f"Permission denied creating directory: {clone_dir}")

    # Extract qualified repo name (owner__repo) to avoid collisions
    repo_name = extract_qualified_repo_name(repo_url)
    target_path = clone_path / repo_name

    # Check if directory already exists
    if target_path.exists():
        if is_git_repository(str(target_path)):
            try:
                # Attempt checkout
                checkout_branch(str(target_path), branch)
                return str(target_path.absolute())
            except Exception:
                # Repo exists but is corrupted (e.g. broken HEAD, incomplete clone).
                # Remove and re-clone below.
                shutil.rmtree(target_path)
        else:
            raise Exception(f"Directory exists but is not a git repository: {target_path}")

    # Clone the repository (idle timeout — keeps running as long as data flows)
    try:
        _run_with_idle_timeout(
            ["git", "clone", authenticated_url, str(target_path)],
            idle_timeout=timeout,
            watch_path=str(target_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # Mask the token in error logs
        error_msg = e.stderr.lower()
        safe_stderr = e.stderr
        if token:
            safe_stderr = safe_stderr.replace(token, "[REDACTED]")

        if 'could not resolve host' in error_msg or 'failed to connect' in error_msg:
            raise Exception(f"Network failure: Unable to connect to {repo_url}")
        elif 'authentication failed' in error_msg or 'permission denied' in error_msg:
            # Provide specific hint based on domain
            hint = ""
            if 'azure' in repo_url: hint = " (For Azure, ensure your PAT has 'Code (Read)' scope)"
            if 'bitbucket' in repo_url: hint = " (For Bitbucket, ensure this is an OAuth token or App Password)"

            raise Exception(f"Authentication failed for {repo_url}. Check your token.{hint}")
        else:
            raise Exception(f"Failed to clone repository: {safe_stderr}")

    except subprocess.TimeoutExpired:
        raise Exception(f"Clone operation timed out for {repo_url}")

    # Checkout the specified branch (skip if clone already landed on it)
    try:
        current = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=str(target_path), capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        current = None

    if current != branch:
        try:
            checkout_branch(str(target_path), branch)
        except Exception:
            # Branch doesn't exist — keep the default branch from clone
            pass

    return str(target_path.absolute())


def get_github_repos(org, token):
    """GitHub API Logic. Tries the orgs endpoint first; falls back to the
    users endpoint when the target is a personal account (orgs returns 404)."""
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    repo_type = "all" if token else "public"
    repo_data = []
    page = 1

    print(f"Fetching GitHub repos for {org} (type={repo_type})...", file=sys.stderr)

    # Try orgs endpoint first
    probe_url = f"https://api.github.com/orgs/{org}/repos?type={repo_type}&per_page=100&page=1"
    probe = requests.get(probe_url, headers=headers)

    if probe.status_code == 404:
        # Fall back to user repos endpoint
        print(f"  Not an org, trying as user account...", file=sys.stderr)
        base_url = f"https://api.github.com/users/{org}/repos?type={repo_type}&per_page=100"
        while True:
            url = f"{base_url}&page={page}"
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                print(f"GitHub Error: {response.status_code} - {response.text}", file=sys.stderr)
                sys.exit(1)
            data = response.json()
            if not data:
                break
            for repo in data:
                repo_data.append({
                    "url": repo['clone_url'],
                    "branch": repo['default_branch']
                })
            page += 1
        return repo_data

    # Orgs endpoint succeeded — continue paginating from probe response
    if probe.status_code != 200:
        print(f"GitHub Error: {probe.status_code} - {probe.text}", file=sys.stderr)
        sys.exit(1)

    data = probe.json()
    for repo in data:
        repo_data.append({
            "url": repo['clone_url'],
            "branch": repo['default_branch']
        })
    if len(data) < 100:
        return repo_data
    page = 2

    while True:
        url = f"https://api.github.com/orgs/{org}/repos?type={repo_type}&per_page=100&page={page}"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"GitHub Error: {response.status_code} - {response.text}", file=sys.stderr)
            sys.exit(1)

        data = response.json()
        if not data: break

        for repo in data:
            repo_data.append({
                "url": repo['clone_url'],
                "branch": repo['default_branch']
            })
        page += 1
    return repo_data


def get_gitlab_repos(group, token):
    """GitLab API Logic"""
    headers = {"PRIVATE-TOKEN": token}
    repo_data = []
    page = 1

    print(f"Fetching GitLab projects for group {group}...", file=sys.stderr)

    while True:
        # Fetches projects from the group. simple=true returns lighter objects.
        # include_subgroups=true is often useful for large orgs.
        url = f"https://gitlab.com/api/v4/groups/{group}/projects?visibility=private&simple=true&per_page=100&page={page}&include_subgroups=true"
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"GitLab Error: {response.status_code} - {response.text}", file=sys.stderr)
            sys.exit(1)

        data = response.json()
        if not data: break

        for repo in data:
            repo_data.append({
                "url": repo['http_url_to_repo'],
                "branch": repo['default_branch']
            })
        page += 1
    return repo_data


def get_azure_repos(org, token):
    """Azure DevOps API Logic"""
    # Azure requires Basic Auth with an empty username and the PAT as password.
    # We must Base64 encode ":{token}"
    auth_str = f":{token}"
    b64_auth = base64.b64encode(auth_str.encode()).decode()

    headers = {"Authorization": f"Basic {b64_auth}"}
    repo_data = []

    print(f"Fetching Azure repos for {org}...", file=sys.stderr)

    # Azure doesn't paginate strictly by page number usually, but for standard list calls
    # it often returns all or uses continuation tokens. We'll try the standard list.
    # Note: Azure URLs are usually https://dev.azure.com/{org}/_apis/git/repositories
    url = f"https://dev.azure.com/{org}/_apis/git/repositories?api-version=6.0"

    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"Azure Error: {response.status_code} - {response.text}", file=sys.stderr)
        sys.exit(1)

    data = response.json()

    # Azure returns a 'value' list
    for repo in data.get('value', []):
        # Azure Default branch is often 'refs/heads/main', we strip the prefix
        full_branch = repo.get('defaultBranch', 'refs/heads/main')
        branch_name = full_branch.replace('refs/heads/', '')

        repo_data.append({
            "url": repo['webUrl'],  # Or remoteUrl, usually webUrl works for cloning if .git is appended or supported
            "branch": branch_name
        })

    return repo_data


def get_bitbucket_repos(workspace, token):
    """Bitbucket Cloud API Logic"""
    # Assuming 'token' is an OAuth Access Token (Bearer).
    # If using App Password, the user must provide "username:app_password" string as token
    # and we would assume Basic Auth. Here we assume Bearer for simplicity.
    headers = {"Authorization": f"Bearer {token}"}
    repo_data = []

    print(f"Fetching Bitbucket repos for workspace {workspace}...", file=sys.stderr)

    url = f"https://api.bitbucket.org/2.0/repositories/{workspace}?q=is_private=true"

    while url:
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            print(f"Bitbucket Error: {response.status_code} - {response.text}", file=sys.stderr)
            sys.exit(1)

        data = response.json()

        for repo in data.get('values', []):
            # Bitbucket returns a list of clone links (https and ssh)
            clone_links = repo.get('links', {}).get('clone', [])
            https_link = next((l['href'] for l in clone_links if l['name'] == 'https'), None)

            # Bitbucket main branch info structure
            main_branch = repo.get('mainbranch', {}).get('name', 'master')

            if https_link:
                repo_data.append({
                    "url": https_link,
                    "branch": main_branch
                })

        # Pagination: Bitbucket provides a 'next' URL
        url = data.get('next')

    return repo_data


# --- Main Dispatcher ---

def get_repo_details(git_type, org, token):
    if git_type == 'github':
        return get_github_repos(org, token)
    elif git_type == 'gitlab':
        return get_gitlab_repos(org, token)
    elif git_type == 'azure':
        return get_azure_repos(org, token)
    elif git_type == 'bitbucket':
        return get_bitbucket_repos(org, token)
    else:
        print(f"Unsupported git type: {git_type}", file=sys.stderr)
        sys.exit(1)
