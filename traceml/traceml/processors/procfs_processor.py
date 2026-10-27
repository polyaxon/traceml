import os
import platform
import resource
from typing import Dict, List, Optional, Set

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


def _cgroup_v2_dir(proc_root: str, cgroup_root: str) -> Optional[str]:
    if not os.path.exists(os.path.join(cgroup_root, "cgroup.controllers")):
        return None
    text = _read_text(os.path.join(proc_root, "self", "cgroup")) or ""
    for line in text.splitlines():
        if line.startswith("0::"):
            candidate = os.path.join(cgroup_root, line[3:].strip().lstrip("/"))
            if os.path.exists(os.path.join(candidate, "memory.current")):
                return candidate
    # Inside a container the path in /proc/self/cgroup is the host path,
    # but the mount only shows the container's own cgroup at its root.
    if os.path.exists(os.path.join(cgroup_root, "memory.current")):
        return cgroup_root
    return None


def _query_cgroup_v2(cg_dir: str) -> Dict:
    results = {}  # type: Dict[str, float]
    current = _read_int(os.path.join(cg_dir, "memory.current"))
    if current is not None:
        results["cgroup_memory_current"] = current
    maximum = _read_int(os.path.join(cg_dir, "memory.max"))
    if maximum is not None:
        results["cgroup_memory_max"] = maximum
        if current is not None and maximum:
            results["cgroup_memory_percent"] = round(100.0 * current / maximum, 2)
    stat = _read_kv(os.path.join(cg_dir, "memory.stat"))
    if "shmem" in stat:
        results["cgroup_memory_shmem"] = stat["shmem"]
    events = _read_kv(os.path.join(cg_dir, "memory.events"))
    if "oom_kill" in events:
        results["cgroup_oom_kill"] = events["oom_kill"]

    cpu = _read_kv(os.path.join(cg_dir, "cpu.stat"))
    if "nr_throttled" in cpu:
        results["cgroup_cpu_nr_throttled"] = cpu["nr_throttled"]
    if "throttled_usec" in cpu:
        results["cgroup_cpu_throttled_usec"] = cpu["throttled_usec"]
    cpu_max = (_read_text(os.path.join(cg_dir, "cpu.max")) or "").split()
    if len(cpu_max) == 2 and cpu_max[0] != "max":
        try:
            results["cgroup_cpu_quota_cpus"] = round(
                int(cpu_max[0]) / int(cpu_max[1]), 3
            )
        except (ValueError, ZeroDivisionError):
            pass
    return _only_if_limited(results)


def _cgroup_v1_dir(proc_root: str, cgroup_root: str, controller: str) -> Optional[str]:
    base = os.path.join(cgroup_root, controller)
    if not os.path.isdir(base):
        return None
    text = _read_text(os.path.join(proc_root, "self", "cgroup")) or ""
    for line in text.splitlines():
        parts = line.split(":")
        if len(parts) == 3 and controller in parts[1].split(","):
            candidate = os.path.join(base, parts[2].strip().lstrip("/"))
            if os.path.isdir(candidate):
                return candidate
    return base


def _query_cgroup_v1(proc_root: str, cgroup_root: str) -> Dict:
    results = {}  # type: Dict[str, float]
    mem_dir = _cgroup_v1_dir(proc_root, cgroup_root, "memory")
    if mem_dir:
        current = _read_int(os.path.join(mem_dir, "memory.usage_in_bytes"))
        if current is not None:
            results["cgroup_memory_current"] = current
        maximum = _read_int(os.path.join(mem_dir, "memory.limit_in_bytes"))
        if maximum is not None and maximum < _V1_UNLIMITED:
            results["cgroup_memory_max"] = maximum
            if current is not None and maximum:
                results["cgroup_memory_percent"] = round(100.0 * current / maximum, 2)
        stat = _read_kv(os.path.join(mem_dir, "memory.stat"))
        if "shmem" in stat:
            results["cgroup_memory_shmem"] = stat["shmem"]
        oom = _read_kv(os.path.join(mem_dir, "memory.oom_control"))
        if "oom_kill" in oom:
            results["cgroup_oom_kill"] = oom["oom_kill"]

    cpu_dir = _cgroup_v1_dir(proc_root, cgroup_root, "cpu")
    if cpu_dir:
        cpu = _read_kv(os.path.join(cpu_dir, "cpu.stat"))
        if "nr_throttled" in cpu:
            results["cgroup_cpu_nr_throttled"] = cpu["nr_throttled"]
        if "throttled_time" in cpu:
            results["cgroup_cpu_throttled_usec"] = cpu["throttled_time"] // 1000
        quota = _read_int(os.path.join(cpu_dir, "cpu.cfs_quota_us"))
        period = _read_int(os.path.join(cpu_dir, "cpu.cfs_period_us"))
        if quota is not None and quota > 0 and period:
            results["cgroup_cpu_quota_cpus"] = round(quota / period, 3)
    return _only_if_limited(results)


def query_cgroup(proc_root: str = PROC_ROOT, cgroup_root: str = CGROUP_ROOT) -> Dict:
    """Memory and CPU accounting of the cgroup that holds the current process.

    This is the pod or container limit, which `psutil.virtual_memory()` does not see.
    """
    cg_dir = _cgroup_v2_dir(proc_root, cgroup_root)
    if cg_dir:
        return _query_cgroup_v2(cg_dir)
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
