import os
import platform
import resource
from typing import Dict, List, Optional, Set, Tuple

from traceml.processors.events_processors import metrics_dict_to_list


PROC_ROOT = "/proc"
CGROUP_ROOT = "/sys/fs/cgroup"
SHM_PATH = "/dev/shm"

# Same key as `polyaxon._env_vars.keys.ENV_KEYS_COLLECT_RESOURCES_DETAILED`,
# duplicated so that this module works with older polyaxon versions.
ENV_KEYS_COLLECT_RESOURCES_DETAILED = "POLYAXON_COLLECT_RESOURCES_DETAILED"

# Logged by default: one metric per failure mode (OOM, cpu throttling,
# full /dev/shm, stuck workers). The remaining metrics are logged when
# `POLYAXON_COLLECT_RESOURCES_DETAILED` is set.
CORE_METRICS = frozenset(
    (
        "cgroup_memory_percent",
        "cgroup_oom_kill",
        "cgroup_cpu_throttled_usec",
        "shm_percent",
        "proc_children_state_d",
    )
)

_KB = 1024
# cgroup v1 reports "no limit" as a very large number instead of "max".
_V1_UNLIMITED = 1 << 60


def can_log_procfs_resources(proc_root: str = PROC_ROOT) -> bool:
    return platform.system() == "Linux" and os.path.isdir(
        os.path.join(proc_root, "self")
    )


def is_detailed_enabled() -> bool:
    value = os.getenv(ENV_KEYS_COLLECT_RESOURCES_DETAILED, "")
    return value.strip().lower() in ("1", "true", "t", "yes", "y", "on")


def select_metrics(results: Dict, detailed: bool) -> Dict:
    """Keeps the core metrics unless `detailed` is set."""
    if detailed:
        return results
    selected = {k: v for k, v in results.items() if k in CORE_METRICS}
    # A cgroup with a cpu quota but no memory limit has no percent to report.
    if "cgroup_memory_percent" not in results and "cgroup_memory_current" in results:
        selected["cgroup_memory_current"] = results["cgroup_memory_current"]
    return selected


def _read_text(path: str) -> Optional[str]:
    try:
        with open(path) as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def _read_int(path: str) -> Optional[int]:
    """Reads a single integer file. Returns None for missing files and for `max`."""
    text = _read_text(path)
    if not text:
        return None
    first = text.split()[0] if text.split() else ""
    try:
        return int(first)
    except ValueError:
        return None


def _read_kv(path: str) -> Dict[str, int]:
    """Reads a `key value` file such as cgroup `memory.stat` or `cpu.stat`."""
    text = _read_text(path)
    results = {}  # type: Dict[str, int]
    if not text:
        return results
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            results[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return results


def _read_status(pid_dir: str) -> Dict[str, object]:
    """Reads `/proc/<pid>/status`. Memory values are returned in bytes."""
    text = _read_text(os.path.join(pid_dir, "status"))
    results = {}  # type: Dict[str, object]
    if not text:
        return results
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip()
        if key == "State":
            results[key] = value[:1]
        elif key in ("Threads", "Pid", "PPid"):
            try:
                results[key] = int(value)
            except ValueError:
                pass
        elif key in ("VmRSS", "RssAnon", "RssFile", "RssShmem", "VmSize"):
            parts = value.split()
            try:
                results[key] = int(parts[0]) * _KB
            except (IndexError, ValueError):
                pass
    return results


def _children_pids(pid_dir: str) -> Set[int]:
    """Collects the direct children of a process from `/proc/<pid>/task/*/children`."""
    pids = set()  # type: Set[int]
    task_dir = os.path.join(pid_dir, "task")
    try:
        tids = os.listdir(task_dir)
    except OSError:
        return pids
    for tid in tids:
        text = _read_text(os.path.join(task_dir, tid, "children"))
        if not text:
            continue
        for item in text.split():
            try:
                pids.add(int(item))
            except ValueError:
                continue
    return pids


def _fd_stats(pid_dir: str, shm_path: str) -> Dict[str, int]:
    fd_dir = os.path.join(pid_dir, "fd")
    try:
        names = os.listdir(fd_dir)
    except OSError:
        return {}
    shm = 0
    for name in names:
        try:
            target = os.readlink(os.path.join(fd_dir, name))
        except OSError:
            continue
        if target.startswith(shm_path):
            shm += 1
    return {"proc_fd_count": len(names), "proc_fd_shm": shm}


def _fd_limit() -> Optional[int]:
    try:
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError):
        return None
    if soft == resource.RLIM_INFINITY:
        return None
    return int(soft)


def query_process(proc_root: str = PROC_ROOT, shm_path: str = SHM_PATH) -> Dict:
    """Memory, file descriptors and children of the current process."""
    results = {}  # type: Dict[str, int]
    self_dir = os.path.join(proc_root, "self")
    status = _read_status(self_dir)
    for key, name in (
        ("RssAnon", "proc_rss_anon"),
        ("RssFile", "proc_rss_file"),
        ("RssShmem", "proc_rss_shmem"),
        ("Threads", "proc_threads"),
    ):
        if key in status:
            results[name] = status[key]  # type: ignore

    results.update(_fd_stats(self_dir, shm_path=shm_path))
    fd_limit = _fd_limit()
    if fd_limit is not None:
        results["proc_fd_limit"] = fd_limit

    children = _children_pids(self_dir)
    anon = shmem = state_d = 0
    for pid in children:
        child = _read_status(os.path.join(proc_root, str(pid)))
        anon += child.get("RssAnon", 0)  # type: ignore
        shmem += child.get("RssShmem", 0)  # type: ignore
        if child.get("State") == "D":
            state_d += 1
    results["proc_children"] = len(children)
    results["proc_children_rss_anon"] = anon
    results["proc_children_rss_shmem"] = shmem
    results["proc_children_state_d"] = state_d
    return results


def query_shm(shm_path: str = SHM_PATH) -> Dict:
    """Usage of the shared memory mount, usually a tmpfs at `/dev/shm`."""
    try:
        st = os.statvfs(shm_path)
    except OSError:
        return {}
    total = st.f_blocks * st.f_frsize
    used = (st.f_blocks - st.f_bfree) * st.f_frsize
    results = {"shm_total": total, "shm_used": used}  # type: Dict[str, float]
    if total:
        results["shm_percent"] = round(100.0 * used / total, 2)
    return results


def _only_if_limited(results: Dict) -> Dict:
    """Drops cgroup metrics when the cgroup has neither a memory limit nor a cpu quota.

    Without a limit the cgroup is the whole machine, which psutil already covers.
    """
    if "cgroup_memory_max" in results or "cgroup_cpu_quota_cpus" in results:
        return results
    return {}


def _path_parts(path: str) -> List[str]:
    return [p for p in path.split("/") if p]


def _cgroup_membership(proc_root: str, controller: Optional[str]) -> str:
    """The process's cgroup path from /proc/self/cgroup, relative to the hierarchy root.

    `controller` is None for cgroup v2 (the `0::` entry) and the controller name
    for cgroup v1 (the entry that lists it).
    """
    text = _read_text(os.path.join(proc_root, "self", "cgroup")) or ""
    for line in text.splitlines():
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        is_v2 = controller is None and parts[0] == "0" and parts[1] == ""
        is_v1 = controller is not None and controller in parts[1].split(",")
        if is_v2 or is_v1:
            return parts[2]
    return ""


def _cgroup_mount(
    proc_root: str, cgroup_root: str, controller: Optional[str], membership: str
) -> Tuple[str, str]:
    """(mount point, mount root) of the cgroup hierarchy from /proc/self/mountinfo.

    A mount can expose a subtree of the hierarchy (mount root other than "/"),
    which is what Docker does with the host cgroup namespace, cgroup v1 can
    co-mount controllers under a name such as "cpu,cpuacct", and several mounts
    can match, e.g. a bind-mounted subtree next to the full hierarchy. The
    mount whose root contains `membership` is used; among those the one with
    the shortest root, because it exposes the most ancestors and an ancestor
    limit still applies. Then the first match, then `cgroup_root`.
    """
    candidates = []  # type: List[Tuple[str, str]]
    text = _read_text(os.path.join(proc_root, "self", "mountinfo")) or ""
    for line in text.splitlines():
        before, separator, after = line.partition(" - ")
        mount_fields, fs_fields = before.split(), after.split()
        if not separator or len(mount_fields) < 5 or len(fs_fields) < 3:
            continue
        fs_type, super_options = fs_fields[0], fs_fields[2].split(",")
        is_v2 = controller is None and fs_type == "cgroup2"
        is_v1 = (
            controller is not None
            and fs_type == "cgroup"
            and controller in super_options
        )
        if is_v2 or is_v1:
            # mountinfo escapes spaces as \040.
            candidates.append(
                (
                    mount_fields[4].replace("\\040", " "),
                    mount_fields[3].replace("\\040", " "),
                )
            )
    member_parts = _path_parts(membership)
    containing = [
        (point, root)
        for point, root in candidates
        if member_parts[: len(_path_parts(root))] == _path_parts(root)
    ]
    if containing:
        return min(containing, key=lambda mount: len(_path_parts(mount[1])))
    if candidates:
        return candidates[0]
    if controller is None:
        return cgroup_root, "/"
    return os.path.join(cgroup_root, controller), "/"


def _cgroup_dirs(
    proc_root: str, cgroup_root: str, controller: Optional[str]
) -> List[str]:
    """Directories of the process's cgroup, nearest first, mount point last.

    /proc/self/cgroup is relative to the hierarchy root while the mount starts
    at its mount root, so the mount root is stripped from the path first.
    Levels that do not exist in the mount are skipped; the mount point is
    always included.
    """
    membership = _cgroup_membership(proc_root, controller)
    mount_point, mount_root = _cgroup_mount(
        proc_root, cgroup_root, controller, membership
    )
    parts = _path_parts(membership)
    root_parts = _path_parts(mount_root)
    if ".." in parts:
        # A cgroup outside this cgroup namespace cannot be resolved from here.
        parts = []
    elif parts[: len(root_parts)] == root_parts:
        parts = parts[len(root_parts) :]
    dirs = []  # type: List[str]
    for depth in range(len(parts), 0, -1):
        candidate = os.path.join(mount_point, *parts[:depth])
        if os.path.isdir(candidate):
            dirs.append(candidate)
    dirs.append(mount_point)
    return dirs


def _cgroup_dirs_v2(proc_root: str, cgroup_root: str) -> List[str]:
    """Directories of the process's cgroup v2 hierarchy, or [] without cgroup v2."""
    dirs = _cgroup_dirs(proc_root, cgroup_root, None)
    if not os.path.exists(os.path.join(dirs[-1], "cgroup.controllers")):
        return []
    return dirs


def _cgroup_dirs_v1(proc_root: str, cgroup_root: str, controller: str) -> List[str]:
    """Directories of the process's cgroup v1 `controller` hierarchy, nearest first."""
    dirs = _cgroup_dirs(proc_root, cgroup_root, controller)
    if not os.path.isdir(dirs[-1]):
        return []
    return dirs


def _binding_dir(dirs: List[str], read_limit) -> Tuple[str, Optional[float]]:
    """The directory with the smallest limit, which is the one the kernel enforces.

    `read_limit(dir)` returns the limit of one level or None. Without a limit on
    any level the nearest directory is returned with None.
    """
    best, best_limit = dirs[0], None  # type: str, Optional[float]
    for cg_dir in dirs:
        limit = read_limit(cg_dir)
        if limit is not None and (best_limit is None or limit < best_limit):
            best, best_limit = cg_dir, limit
    return best, best_limit


def _tightest_dir(
    dirs: List[str], read_limit, read_current
) -> Tuple[str, Optional[int], Optional[int]]:
    """(directory, limit, current) of the limited level with the least headroom.

    A parent limit also counts the usage of the process's siblings, so a parent
    with a larger limit can be closer to it than the process's own cgroup, and
    the OOM killer fires at that level first. Levels without a limit do not
    count; a limited level whose usage cannot be read ranks after every level
    whose usage can. Without a limit on any level the nearest directory is
    returned with its usage and None.
    """
    best = None  # type: Optional[Tuple[str, int, Optional[int]]]
    best_key = None  # type: Optional[Tuple[bool, int]]
    for cg_dir in dirs:
        limit = read_limit(cg_dir)
        if limit is None:
            continue
        current = read_current(cg_dir)
        key = (current is None, limit if current is None else limit - current)
        if best_key is None or key < best_key:
            best, best_key = (cg_dir, limit, current), key
    if best is None:
        return dirs[0], None, read_current(dirs[0])
    return best


def _read_cpu_max(cg_dir: str) -> Optional[float]:
    cpu_max = (_read_text(os.path.join(cg_dir, "cpu.max")) or "").split()
    if len(cpu_max) != 2 or cpu_max[0] == "max":
        return None
    try:
        return int(cpu_max[0]) / int(cpu_max[1])
    except (ValueError, ZeroDivisionError):
        return None


def _read_cfs_quota(cg_dir: str) -> Optional[float]:
    quota = _read_int(os.path.join(cg_dir, "cpu.cfs_quota_us"))
    period = _read_int(os.path.join(cg_dir, "cpu.cfs_period_us"))
    if quota is None or quota <= 0 or not period:
        return None
    return quota / period


def _read_v1_memory_limit(cg_dir: str) -> Optional[int]:
    limit = _read_int(os.path.join(cg_dir, "memory.limit_in_bytes"))
    if limit is None or limit >= _V1_UNLIMITED:
        return None
    return limit


def _query_cgroup_v2(dirs: List[str]) -> Dict:
    results = {}  # type: Dict[str, float]
    # Usage is read on the level that is closest to its limit, so percent is
    # what the OOM killer sees. A limit on a parent cgroup is found this way.
    mem_dir, maximum, current = _tightest_dir(
        dirs,
        lambda d: _read_int(os.path.join(d, "memory.max")),
        lambda d: _read_int(os.path.join(d, "memory.current")),
    )
    if current is not None:
        results["cgroup_memory_current"] = current
    if maximum is not None:
        results["cgroup_memory_max"] = maximum
        if current is not None and maximum:
            results["cgroup_memory_percent"] = round(100.0 * current / maximum, 2)
    stat = _read_kv(os.path.join(mem_dir, "memory.stat"))
    if "shmem" in stat:
        results["cgroup_memory_shmem"] = stat["shmem"]
    events = _read_kv(os.path.join(mem_dir, "memory.events"))
    if "oom_kill" in events:
        results["cgroup_oom_kill"] = events["oom_kill"]

    cpu_dir, quota = _binding_dir(dirs, _read_cpu_max)
    cpu = _read_kv(os.path.join(cpu_dir, "cpu.stat"))
    if "nr_throttled" in cpu:
        results["cgroup_cpu_nr_throttled"] = cpu["nr_throttled"]
    if "throttled_usec" in cpu:
        results["cgroup_cpu_throttled_usec"] = cpu["throttled_usec"]
    if quota is not None:
        results["cgroup_cpu_quota_cpus"] = round(quota, 3)
    return _only_if_limited(results)


def _query_cgroup_v1(proc_root: str, cgroup_root: str) -> Dict:
    results = {}  # type: Dict[str, float]
    mem_dirs = _cgroup_dirs_v1(proc_root, cgroup_root, "memory")
    if mem_dirs:
        mem_dir, maximum, current = _tightest_dir(
            mem_dirs,
            _read_v1_memory_limit,
            lambda d: _read_int(os.path.join(d, "memory.usage_in_bytes")),
        )
        if current is not None:
            results["cgroup_memory_current"] = current
        if maximum is not None:
            results["cgroup_memory_max"] = maximum
            if current is not None and maximum:
                results["cgroup_memory_percent"] = round(100.0 * current / maximum, 2)
        stat = _read_kv(os.path.join(mem_dir, "memory.stat"))
        if "shmem" in stat:
            results["cgroup_memory_shmem"] = stat["shmem"]
        oom = _read_kv(os.path.join(mem_dir, "memory.oom_control"))
        if "oom_kill" in oom:
            results["cgroup_oom_kill"] = oom["oom_kill"]

    cpu_dirs = _cgroup_dirs_v1(proc_root, cgroup_root, "cpu")
    if cpu_dirs:
        cpu_dir, quota = _binding_dir(cpu_dirs, _read_cfs_quota)
        cpu = _read_kv(os.path.join(cpu_dir, "cpu.stat"))
        if "nr_throttled" in cpu:
            results["cgroup_cpu_nr_throttled"] = cpu["nr_throttled"]
        if "throttled_time" in cpu:
            results["cgroup_cpu_throttled_usec"] = cpu["throttled_time"] // 1000
        if quota is not None:
            results["cgroup_cpu_quota_cpus"] = round(quota, 3)
    return _only_if_limited(results)


def query_cgroup(proc_root: str = PROC_ROOT, cgroup_root: str = CGROUP_ROOT) -> Dict:
    """Memory and CPU accounting of the cgroup that limits the current process.

    Memory is read on the level, from the process's cgroup up to the root, that
    is closest to its limit; the cpu quota is the smallest one. This is the pod
    or container limit, which `psutil.virtual_memory()` does not see.

    cgroup v1 is tried when cgroup v2 reports no limit: on a systemd hybrid
    host cgroup v2 is mounted without controllers and the limits live in v1.
    """
    dirs = _cgroup_dirs_v2(proc_root, cgroup_root)
    results = _query_cgroup_v2(dirs) if dirs else {}
    if results:
        return results
    return _query_cgroup_v1(proc_root, cgroup_root)


def query_procfs(
    proc_root: str = PROC_ROOT,
    cgroup_root: str = CGROUP_ROOT,
    shm_path: str = SHM_PATH,
) -> Dict:
    results = {}
    results.update(query_process(proc_root=proc_root, shm_path=shm_path))
    results.update(query_shm(shm_path=shm_path))
    results.update(query_cgroup(proc_root=proc_root, cgroup_root=cgroup_root))
    return results


def get_procfs_metrics(detailed: Optional[bool] = None) -> List:
    if detailed is None:
        detailed = is_detailed_enabled()
    return metrics_dict_to_list(select_metrics(query_procfs(), detailed=detailed))
