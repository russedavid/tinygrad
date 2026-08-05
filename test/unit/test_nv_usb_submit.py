import types, unittest
from unittest.mock import MagicMock, patch

from tinygrad.runtime.autogen import nv_570 as nv_gpu
from tinygrad.runtime.ops_nv import GPFifo, NVAllocator, NVCommandQueue, NVDevice, PCIIface, USBIface
from tinygrad.runtime.support.hcq import HCQAllocator, HCQCompiled
from tinygrad.runtime.support.nv.nvdev import NVLifecyclePhase


class RecordingMMIO:
  def __init__(self, events, name): self.events, self.name, self.values = events, name, {}
  def __setitem__(self, index, value):
    self.values[index] = value
    self.events.append((self.name, index, value))
  def __getitem__(self, index):
    self.events.append((f"{self.name}_read", index))
    return self.values[index]


class TestNVUSBSubmission(unittest.TestCase):
  def test_submission_without_pci_transport(self):
    events = []
    dev = types.SimpleNamespace(iface=types.SimpleNamespace(), gpu_mmio=RecordingMMIO(events, "doorbell"),
                                _set_perf_boost=lambda enabled: events.append(("boost", enabled)))
    queue = object.__new__(NVCommandQueue)
    queue.binded_device, queue.hw_page, queue._q = dev, types.SimpleNamespace(va_addr=0x1000), [0, 0]
    gpfifo = GPFifo(RecordingMMIO(events, "ring"), RecordingMMIO(events, "gpput"), 16, token=0x1234)

    with patch("tinygrad.runtime.ops_nv.System.memory_barrier", MagicMock()): queue._submit_to_gpfifo(dev, gpfifo)

    self.assertEqual(events[0], ("boost", True))
    self.assertEqual([event[0] for event in events], ["boost", "ring", "gpput", "doorbell"])
    queue.binded_device = None

  def test_flushes_bar_writes_before_and_after_doorbell(self):
    events = []
    pci_dev = types.SimpleNamespace(flush_writes=lambda: events.append(("flush",)))
    dev = types.SimpleNamespace(iface=types.SimpleNamespace(pci_dev=pci_dev), gpu_mmio=RecordingMMIO(events, "doorbell"),
                                _set_perf_boost=lambda enabled: None)
    queue = object.__new__(NVCommandQueue)
    queue.binded_device, queue.hw_page, queue._q = dev, types.SimpleNamespace(va_addr=0x1000), [0, 0, 0, 0]
    gpfifo = GPFifo(RecordingMMIO(events, "ring"), RecordingMMIO(events, "gpput"), 16, token=0x1234)

    with patch("tinygrad.runtime.ops_nv.System.memory_barrier", MagicMock()): queue._submit_to_gpfifo(dev, gpfifo)

    self.assertEqual([event[0] for event in events], ["ring", "gpput", "flush", "doorbell", "flush"])
    self.assertEqual(events[-2], ("doorbell", 0x90 // 4, 0x1234))
    queue.binded_device = None

  def test_submission_uses_transport_doorbell_token(self):
    events = []
    pci_dev = types.SimpleNamespace(flush_writes=lambda: events.append(("flush",)))
    iface = types.SimpleNamespace(pci_dev=pci_dev, gpfifo_submit_token=lambda token: token & 0x7ff)
    dev = types.SimpleNamespace(iface=iface, gpu_mmio=RecordingMMIO(events, "doorbell"), _set_perf_boost=lambda enabled: None)
    queue = object.__new__(NVCommandQueue)
    queue.binded_device, queue.hw_page, queue._q = dev, types.SimpleNamespace(va_addr=0x1000), [0, 0]
    gpfifo = GPFifo(RecordingMMIO(events, "ring"), RecordingMMIO(events, "gpput"), 16, token=0x2001234)

    with patch("tinygrad.runtime.ops_nv.System.memory_barrier", MagicMock()): queue._submit_to_gpfifo(dev, gpfifo)

    self.assertIn(("doorbell", 0x90 // 4, 0x234), events)
    queue.binded_device = None

  def test_verifies_bar_writes_before_doorbell(self):
    events = []
    pci_dev = types.SimpleNamespace(flush_writes=lambda: events.append(("flush",)), verify_bar_writes=True)
    dev = types.SimpleNamespace(iface=types.SimpleNamespace(pci_dev=pci_dev), gpu_mmio=RecordingMMIO(events, "doorbell"),
                                _set_perf_boost=lambda enabled: None)
    queue = object.__new__(NVCommandQueue)
    queue.binded_device, queue.hw_page, queue._q = dev, types.SimpleNamespace(va_addr=0x1000), [0, 0]
    gpfifo = GPFifo(RecordingMMIO(events, "ring"), RecordingMMIO(events, "gpput"), 16, token=0x1234)

    with patch("tinygrad.runtime.ops_nv.System.memory_barrier", MagicMock()): queue._submit_to_gpfifo(dev, gpfifo)

    self.assertEqual([event[0] for event in events],
                     ["ring", "gpput", "flush", "ring_read", "gpput_read", "doorbell", "flush"])
    queue.binded_device = None

  def test_ampere_usb_uses_internal_runlist_doorbell(self):
    iface = object.__new__(USBIface)
    runlist_mmio = object()
    iface.dev = types.SimpleNamespace(subdevice=0xC1000002)
    iface.dev_impl = types.SimpleNamespace(chip_name="GA102")
    iface.pci_dev = types.SimpleNamespace(map_bar=MagicMock(return_value=runlist_mmio))

    def rm_control(obj, cmd, params):
      self.assertEqual(obj, iface.dev.subdevice)
      if cmd == 0x20800179:
        self.assertEqual(params.engineList[0], 1)
        params.runlistPriBase[0], params.runlistId[0] = 0x2A0000, 3
      else: self.fail(f"unexpected RM control {cmd:#x}")
      return params
    iface.rm_control = rm_control

    usermode, mapped = iface.setup_usermode()

    self.assertEqual((usermode, mapped), (0xCE000000, runlist_mmio))
    self.assertEqual((iface.runlist_pri_base, iface.runlist_id), (0x2A0000, 3))
    iface.pci_dev.map_bar.assert_called_once_with(bar=0, fmt='I', off=0x2A0000, size=0x1000)
    self.assertEqual(iface.gpfifo_submit_token(0x1205), 0x205)

  def test_blackwell_usb_keeps_virtual_function_doorbell(self):
    iface = object.__new__(USBIface)
    mmio = object()
    iface.dev_impl = types.SimpleNamespace(chip_name="GB202")
    iface.pci_dev = types.SimpleNamespace(map_bar=MagicMock(return_value=mmio))

    usermode, mapped = iface.setup_usermode()

    self.assertEqual((usermode, mapped), (0xCE000000, mmio))
    iface.pci_dev.map_bar.assert_called_once_with(bar=0, fmt='I', off=0xBB0000, size=0x10000)
    self.assertEqual(iface.gpfifo_submit_token(0x12345678), 0x12345678)

  def test_usb_synchronize_clears_boost_and_submission_restores_it(self):
    iface = types.SimpleNamespace(low_power_idle=True, rm_control=MagicMock())
    dev = object.__new__(NVDevice)
    dev.iface, dev.subdevice, dev._perf_boosted = iface, 0xC1000002, True

    with patch.object(HCQCompiled, "synchronize") as synchronize: dev.synchronize(timeout=123)

    synchronize.assert_called_once()
    obj, cmd, params = iface.rm_control.call_args.args
    self.assertEqual((obj, cmd, params.duration), (dev.subdevice, nv_gpu.NV2080_CTRL_CMD_PERF_BOOST, 0))
    self.assertEqual(params.flags & 0x3, nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CMD_CLEAR)
    self.assertFalse(dev._perf_boosted)

    dev._set_perf_boost(True)
    _, _, params = iface.rm_control.call_args.args
    self.assertEqual(params.flags & 0x3, nv_gpu.NV2080_CTRL_PERF_BOOST_FLAGS_CMD_BOOST_TO_MAX)
    self.assertEqual(params.duration, nv_gpu.NV2080_CTRL_PERF_BOOST_DURATION_INFINITE)
    self.assertTrue(dev._perf_boosted)

    dev._set_perf_boost(True)
    self.assertEqual(iface.rm_control.call_count, 2)

  def test_usb_copyout_returns_to_idle(self):
    dev = types.SimpleNamespace(iface=types.SimpleNamespace(low_power_idle=True), _set_perf_boost=MagicMock())
    allocator = object.__new__(NVAllocator)
    allocator.dev = dev

    with patch.object(HCQAllocator, "_copyout") as copyout: allocator._copyout(MagicMock(), MagicMock())

    copyout.assert_called_once()
    dev._set_perf_boost.assert_called_once_with(False)

  def test_finalize_is_idempotent(self):
    dev = object.__new__(NVDevice)
    dev._finalized = False

    with patch.object(HCQCompiled, "finalize") as finalize:
      dev.finalize()
      dev.finalize()

    finalize.assert_called_once_with()
    self.assertTrue(dev._finalized)

  def test_process_setup_failure_finalizes_interface_without_retry(self):
    iface = object.__new__(USBIface)
    iface.pci_dev = types.SimpleNamespace(gsp_full_teardown=True)
    iface.dev_impl, iface.device_fini = MagicMock(), MagicMock()
    iface.dev_impl._run_phase.side_effect = lambda phase, state, fn: fn()
    failure = RuntimeError("channel allocation failed")

    with patch.object(NVDevice, "_select_iface", return_value=iface), patch.object(NVDevice, "_init_device", side_effect=failure) as init_device:
      with self.assertRaisesRegex(RuntimeError, "channel allocation failed") as raised: NVDevice("NV")

    self.assertIs(raised.exception, failure)
    init_device.assert_called_once_with()
    iface.device_fini.assert_called_once_with()

  def test_interface_setup_failure_finalizes_booted_gsp(self):
    iface = object.__new__(PCIIface)
    iface.dev_impl = MagicMock()
    failure = RuntimeError("root allocation failed")
    iface._init_nvd = MagicMock(side_effect=failure)

    with self.assertRaisesRegex(RuntimeError, "root allocation failed") as raised: iface._init_nvd_or_cleanup()

    self.assertIs(raised.exception, failure)
    iface.dev_impl.initialize.assert_called_once_with()
    iface._init_nvd.assert_called_once_with()
    iface.dev_impl.fini.assert_called_once_with(None)

  def test_interface_boot_failure_finalizes_partially_initialized_device(self):
    iface = object.__new__(PCIIface)
    failure = RuntimeError("Booter Load failed")
    iface.dev_impl = MagicMock()
    iface.dev_impl.initialize.side_effect = failure
    iface._init_nvd = MagicMock()

    with self.assertRaisesRegex(RuntimeError, "Booter Load failed") as raised: iface._init_nvd_or_cleanup()

    self.assertIs(raised.exception, failure)
    iface._init_nvd.assert_not_called()
    iface.dev_impl.fini.assert_called_once_with(None)

  def test_unclean_recovery_failure_is_not_retried_by_initialize_cleanup(self):
    iface = object.__new__(PCIIface)
    failure = RuntimeError("queue attachment failed")
    iface.dev_impl = MagicMock(lifecycle_phase=NVLifecyclePhase.UNCLEAN_RECOVERY)
    iface.dev_impl.initialize.side_effect = failure
    iface._init_nvd = MagicMock()

    with self.assertRaisesRegex(RuntimeError, "queue attachment failed") as raised: iface._init_nvd_or_cleanup()

    self.assertIs(raised.exception, failure)
    iface._init_nvd.assert_not_called()
    iface.dev_impl.fini.assert_not_called()

if __name__ == "__main__": unittest.main()
