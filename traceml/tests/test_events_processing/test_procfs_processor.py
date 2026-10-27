from mock import patch
import os
import pytest
import shutil
import tempfile

from polyaxon._utils.test_utils import BaseTestCase
from traceml.processors.procfs_processor import (
    CORE_METRICS,
    ENV_KEYS_COLLECT_RESOURCES_DETAILED,
    can_log_procfs_resources,
    get_procfs_metrics,
    is_detailed_enabled,
    query_cgroup,
    query_process,
    query_procfs,
    query_shm,
    select_metrics,
)


def _write(path: str, text: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


STATUS_MAIN = """Name:\tpython3
State:\tS (sleeping)
Pid:\t4021
PPid:\t4000
Threads:\t3
VmSize:\t  311584 kB
VmRSS:\t  292480 kB
RssAnon:\t  200000 kB
RssFile:\t   30000 kB
RssShmem:\t   62480 kB
"""

STATUS_WORKER_S = """Name:\tpython3
State:\tS (sleeping)
Pid:\t4101
PPid:\t4021
Threads:\t1
RssAnon:\t   50000 kB
RssFile:\t   10000 kB
RssShmem:\t   38000 kB
"""

STATUS_WORKER_D = """Name:\tpython3
State:\tD (disk sleep)
Pid:\t4102
PPid:\t4021
Threads:\t1
RssAnon:\t   70000 kB
RssFile:\t   10000 kB
RssShmem:\t   38000 kB
"""


@pytest.mark.processors_mark
class TestProcfsProcessor(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.root = tempfile.mkdtemp()
        self.proc = os.path.join(self.root, "proc")
        self.cgroup = os.path.join(self.root, "cgroup")
        self.shm = os.path.join(self.root, "shm")
        os.makedirs(self.shm)

        # Main process and two children, one of them in state D.
        _write(os.path.join(self.proc, "self", "status"), STATUS_MAIN)
        _write(
            os.path.join(self.proc, "self", "task", "4021", "children"), "4101 4102 "
        )
        _write(os.path.join(self.proc, "4101", "status"), STATUS_WORKER_S)
        _write(os.path.join(self.proc, "4102", "status"), STATUS_WORKER_D)

        fd_dir = os.path.join(self.proc, "self", "fd")
        os.makedirs(fd_dir)
        os.symlink("/dev/null", os.path.join(fd_dir, "0"))
        os.symlink("pipe:[22529]", os.path.join(fd_dir, "1"))
        os.symlink("socket:[22530]", os.path.join(fd_dir, "2"))
        os.symlink(self.shm + "/torch_4101_1 (deleted)", os.path.join(fd_dir, "3"))
        os.symlink(self.shm + "/torch_4101_2 (deleted)", os.path.join(fd_dir, "4"))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)
        super().tearDown()

    def _make_cgroup_v2(self, memory_max="2147483648", cpu_max="50000 100000"):
        _write(os.path.join(self.proc, "self", "cgroup"), "0::/lab/box\n")
        cg = os.path.join(self.cgroup, "lab", "box")
        _write(os.path.join(self.cgroup, "cgroup.controllers"), "cpu memory pids\n")
        _write(os.path.join(cg, "memory.current"), "1073741824\n")
        _write(os.path.join(cg, "memory.max"), memory_max + "\n")
        _write(
            os.path.join(cg, "memory.stat"),
            "anon 900000000\nfile 1000\nshmem 173741824\n",
        )
        _write(
            os.path.join(cg, "memory.events"),
            "low 0\nhigh 0\nmax 53\noom 1\noom_kill 1\n",
        )
        _write(
            os.path.join(cg, "cpu.stat"),
            "usage_usec 100\nnr_periods 60\nnr_throttled 56\nthrottled_usec 19317859\n",
        )
        _write(os.path.join(cg, "cpu.max"), cpu_max + "\n")

    def _make_cgroup_v1(self):
        _write(
            os.path.join(self.proc, "self", "cgroup"),
            "12:memory:/kubepods/pod1\n3:cpu,cpuacct:/kubepods/pod1\n",
        )
        mem = os.path.join(self.cgroup, "memory", "kubepods", "pod1")
        cpu = os.path.join(self.cgroup, "cpu", "kubepods", "pod1")
        _write(os.path.join(mem, "memory.usage_in_bytes"), "536870912\n")
        _write(os.path.join(mem, "memory.limit_in_bytes"), "1073741824\n")
        _write(os.path.join(mem, "memory.stat"), "cache 1\nrss 2\nshmem 67108864\n")
        _write(
            os.path.join(mem, "memory.oom_control"),
            "oom_kill_disable 0\nunder_oom 0\noom_kill 2\n",
        )
        _write(
            os.path.join(cpu, "cpu.stat"),
            "nr_periods 10\nnr_throttled 4\nthrottled_time 2000000\n",
        )
        _write(os.path.join(cpu, "cpu.cfs_quota_us"), "200000\n")
        _write(os.path.join(cpu, "cpu.cfs_period_us"), "100000\n")

    def test_can_log_is_bool(self):
        assert can_log_procfs_resources() in (True, False)
        assert can_log_procfs_resources(proc_root=self.root) is False

    def test_process_memory_and_fds(self):
        results = query_process(proc_root=self.proc, shm_path=self.shm)
        assert results["proc_rss_anon"] == 200000 * 1024
        assert results["proc_rss_file"] == 30000 * 1024
        assert results["proc_rss_shmem"] == 62480 * 1024
        assert results["proc_threads"] == 3
        assert results["proc_fd_count"] == 5
        assert results["proc_fd_shm"] == 2
        assert results["proc_fd_limit"] > 0

    def test_process_children(self):
        results = query_process(proc_root=self.proc, shm_path=self.shm)
        assert results["proc_children"] == 2
        assert results["proc_children_rss_anon"] == (50000 + 70000) * 1024
        assert results["proc_children_rss_shmem"] == (38000 + 38000) * 1024
        assert results["proc_children_state_d"] == 1

    def test_process_without_children_file(self):
        shutil.rmtree(os.path.join(self.proc, "self", "task"))
        results = query_process(proc_root=self.proc, shm_path=self.shm)
        assert results["proc_children"] == 0
        assert results["proc_children_rss_anon"] == 0
        assert results["proc_children_rss_shmem"] == 0
        assert results["proc_children_state_d"] == 0

    def test_shm(self):
        results = query_shm(shm_path=self.shm)
        assert results["shm_total"] > 0
        assert 0 <= results["shm_used"] <= results["shm_total"]
        assert 0 <= results["shm_percent"] <= 100

    def test_shm_missing_path(self):
        assert query_shm(shm_path=os.path.join(self.root, "nope")) == {}

    def test_cgroup_v2(self):
        self._make_cgroup_v2()
        results = query_cgroup(proc_root=self.proc, cgroup_root=self.cgroup)
        assert results["cgroup_memory_current"] == 1073741824
        assert results["cgroup_memory_max"] == 2147483648
        assert results["cgroup_memory_percent"] == 50.0
        assert results["cgroup_memory_shmem"] == 173741824
        assert results["cgroup_oom_kill"] == 1
        assert results["cgroup_cpu_nr_throttled"] == 56
        assert results["cgroup_cpu_throttled_usec"] == 19317859
        assert results["cgroup_cpu_quota_cpus"] == 0.5

    def test_cgroup_v2_unlimited(self):
        # No memory limit and no cpu quota: the cgroup is the whole machine.
        self._make_cgroup_v2(memory_max="max", cpu_max="max 100000")
        assert query_cgroup(proc_root=self.proc, cgroup_root=self.cgroup) == {}

    def test_cgroup_v2_cpu_quota_only(self):
        self._make_cgroup_v2(memory_max="max")
        results = query_cgroup(proc_root=self.proc, cgroup_root=self.cgroup)
        assert results["cgroup_cpu_quota_cpus"] == 0.5
        assert results["cgroup_memory_current"] == 1073741824
        assert "cgroup_memory_max" not in results
        assert "cgroup_memory_percent" not in results

    def test_cgroup_v1_unlimited(self):
        self._make_cgroup_v1()
        mem = os.path.join(self.cgroup, "memory", "kubepods", "pod1")
        cpu = os.path.join(self.cgroup, "cpu", "kubepods", "pod1")
        _write(os.path.join(mem, "memory.limit_in_bytes"), "9223372036854771712\n")
        _write(os.path.join(cpu, "cpu.cfs_quota_us"), "-1\n")
        assert query_cgroup(proc_root=self.proc, cgroup_root=self.cgroup) == {}

    def test_cgroup_v2_container_root_fallback(self):
        # Inside a container /proc/self/cgroup shows the host path, but the
        # mount only exposes the container's own cgroup at its root.
        self._make_cgroup_v2()
        _write(
            os.path.join(self.proc, "self", "cgroup"), "0::/kubepods/burstable/pod-x\n"
        )
        for name in ("memory.current", "memory.max"):
            shutil.move(
                os.path.join(self.cgroup, "lab", "box", name),
                os.path.join(self.cgroup, name),
            )
        results = query_cgroup(proc_root=self.proc, cgroup_root=self.cgroup)
        assert results["cgroup_memory_current"] == 1073741824
        assert results["cgroup_memory_percent"] == 50.0

    def test_cgroup_v1(self):
        self._make_cgroup_v1()
        results = query_cgroup(proc_root=self.proc, cgroup_root=self.cgroup)
        assert results["cgroup_memory_current"] == 536870912
        assert results["cgroup_memory_max"] == 1073741824
        assert results["cgroup_memory_percent"] == 50.0
        assert results["cgroup_memory_shmem"] == 67108864
        assert results["cgroup_oom_kill"] == 2
        assert results["cgroup_cpu_nr_throttled"] == 4
        assert results["cgroup_cpu_throttled_usec"] == 2000
        assert results["cgroup_cpu_quota_cpus"] == 2.0

    def test_cgroup_v2_limit_on_parent(self):
        # Kubernetes sets the pod limit on the parent; the container is a child.
        self._make_cgroup_v2(memory_max="max", cpu_max="max 100000")
        lab = os.path.join(self.cgroup, "lab")
        _write(os.path.join(lab, "memory.max"), "2147483648\n")
        _write(os.path.join(lab, "memory.current"), "1610612736\n")
        _write(os.path.join(lab, "memory.events"), "oom_kill 3\n")
        _write(os.path.join(lab, "cpu.max"), "50000 100000\n")
        _write(os.path.join(lab, "cpu.stat"), "nr_throttled 7\nthrottled_usec 500\n")
        results = query_cgroup(proc_root=self.proc, cgroup_root=self.cgroup)
        assert results["cgroup_memory_max"] == 2147483648
        assert results["cgroup_memory_current"] == 1610612736
        assert results["cgroup_memory_percent"] == 75.0
        assert results["cgroup_oom_kill"] == 3
        assert results["cgroup_cpu_quota_cpus"] == 0.5
        assert results["cgroup_cpu_nr_throttled"] == 7

    def test_cgroup_v2_smallest_limit_wins(self):
        self._make_cgroup_v2(memory_max="1073741824")
        _write(os.path.join(self.cgroup, "lab", "memory.max"), "2147483648\n")
        _write(os.path.join(self.cgroup, "lab", "memory.current"), "1\n")
        results = query_cgroup(proc_root=self.proc, cgroup_root=self.cgroup)
        assert results["cgroup_memory_max"] == 1073741824
        assert results["cgroup_memory_current"] == 1073741824

    def test_cgroup_v1_limit_on_parent(self):
        self._make_cgroup_v1()
        pod = os.path.join(self.cgroup, "memory", "kubepods", "pod1")
        parent = os.path.join(self.cgroup, "memory", "kubepods")
        _write(os.path.join(pod, "memory.limit_in_bytes"), "9223372036854771712\n")
        _write(os.path.join(parent, "memory.limit_in_bytes"), "1073741824\n")
        _write(os.path.join(parent, "memory.usage_in_bytes"), "805306368\n")
        results = query_cgroup(proc_root=self.proc, cgroup_root=self.cgroup)
        assert results["cgroup_memory_max"] == 1073741824
        assert results["cgroup_memory_current"] == 805306368
        assert results["cgroup_memory_percent"] == 75.0

    def test_no_cgroup(self):
        assert query_cgroup(proc_root=self.proc, cgroup_root=self.cgroup) == {}

    def test_query_procfs_values_are_numeric(self):
        self._make_cgroup_v2()
        results = query_procfs(
            proc_root=self.proc, cgroup_root=self.cgroup, shm_path=self.shm
        )
        assert results
        for key, value in results.items():
            assert isinstance(value, (int, float)), key

    def test_select_metrics_core(self):
        self._make_cgroup_v2()
        results = query_procfs(
            proc_root=self.proc, cgroup_root=self.cgroup, shm_path=self.shm
        )
        assert len(results) > len(CORE_METRICS)
        core = select_metrics(results, detailed=False)
        assert set(core) == set(CORE_METRICS)
        assert core["cgroup_memory_percent"] == 50.0
        assert core["proc_children_state_d"] == 1

    def test_select_metrics_core_falls_back_to_memory_current(self):
        self._make_cgroup_v2(memory_max="max")
        results = query_procfs(
            proc_root=self.proc, cgroup_root=self.cgroup, shm_path=self.shm
        )
        core = select_metrics(results, detailed=False)
        assert "cgroup_memory_percent" not in core
        assert core["cgroup_memory_current"] == 1073741824

    def test_select_metrics_detailed(self):
        self._make_cgroup_v2()
        results = query_procfs(
            proc_root=self.proc, cgroup_root=self.cgroup, shm_path=self.shm
        )
        assert select_metrics(results, detailed=True) == results

    def test_is_detailed_enabled(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_KEYS_COLLECT_RESOURCES_DETAILED, None)
            assert is_detailed_enabled() is False
        for value in ("true", "True", "1", "yes"):
            with patch.dict(os.environ, {ENV_KEYS_COLLECT_RESOURCES_DETAILED: value}):
                assert is_detailed_enabled() is True
        for value in ("false", "0", "", "no"):
            with patch.dict(os.environ, {ENV_KEYS_COLLECT_RESOURCES_DETAILED: value}):
                assert is_detailed_enabled() is False

    def test_get_procfs_metrics_events(self):
        self._make_cgroup_v2()
        results = query_procfs(
            proc_root=self.proc, cgroup_root=self.cgroup, shm_path=self.shm
        )
        with patch(
            "traceml.processors.procfs_processor.query_procfs", return_value=results
        ):
            core = get_procfs_metrics(detailed=False)
            assert {e.name for e in core} == set(CORE_METRICS)
            detailed = get_procfs_metrics(detailed=True)
            assert {e.name for e in detailed} == set(results)
            for event in detailed:
                assert event.event.metric is not None
                assert event.name.startswith(("proc_", "shm_", "cgroup_"))
            with patch.dict(os.environ, {ENV_KEYS_COLLECT_RESOURCES_DETAILED: "true"}):
                assert len(get_procfs_metrics()) == len(results)
            with patch.dict(os.environ, {ENV_KEYS_COLLECT_RESOURCES_DETAILED: "false"}):
                assert len(get_procfs_metrics()) == len(CORE_METRICS)
