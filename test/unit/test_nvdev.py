import ctypes, types, unittest
from unittest.mock import MagicMock, call, patch

from tinygrad.runtime.autogen import nv_regs
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.memory import AddrSpace, MemoryManager, TLSFAllocator
from tinygrad.runtime.support.nv.ip import NV_FLCN, NVGspLifecycleState, find_vbios_rom_offset
from tinygrad.runtime.support.nv.nvdev import (NVDev, NVLifecycleError, NVLifecyclePhase, NVLifecycleState, NVMemoryManager,
                                               NVPageTableEntry, NVReg)


class StaticDecoder:
  def __init__(self, fields): self.fields = fields
  def decode(self, value): return self.fields


class TestMemoryManagerAllocation(unittest.TestCase):
  def test_contiguous_valloc_can_skip_zeroing(self):
    mm = object.__new__(MemoryManager)
    mm.alloc_vaddr, mm.palloc, mm.map_range = MagicMock(return_value=0x100000), MagicMock(return_value=0x200000), MagicMock()
    mapping = mm.map_range.return_value

    self.assertIs(mm.valloc(0x1800, contiguous=True, cpu_visible=True, zero=False), mapping)

    mm.palloc.assert_called_once_with(0x2000, zero=False, cpu_visible=True)
    mm.map_range.assert_called_once_with(0x100000, 0x2000, [(0x200000, 0x2000)], aspace=AddrSpace.PHYS, uncached=False)

class TestNVBootMemory(unittest.TestCase):
  @staticmethod
  def make_dev(boot_mem_in_vram:bool):
    dev = object.__new__(NVDev)
    dev.large_bar = False
    dev.mm = types.SimpleNamespace(palloc=MagicMock(return_value=0x123000))
    dev.vram = MagicMock()
    dev.vram.view.return_value = MagicMock()
    dev.pci_dev = MagicMock(boot_mem_in_vram=boot_mem_in_vram)
    dev.pci_dev.bar_info.return_value = (0x800000000, 0x600000000)
    return dev

  def test_usb_vram_boot_memory_reports_gpu_offsets(self):
    dev = self.make_dev(True)

    _, paddr, paddrs = dev._alloc_boot_mem(0x1800)

    self.assertEqual(paddr, 0x123000)
    self.assertEqual(paddrs, [0x123000, 0x124000])
    dev.mm.palloc.assert_called_once_with(0x2000, boot=False, cpu_visible=True)
    dev.pci_dev.bar_info.assert_not_called()

  def test_normal_vram_boot_memory_reports_bar_addresses(self):
    dev = self.make_dev(False)

    _, paddr, paddrs = dev._alloc_boot_mem(0x1800, sysmem=False)

    self.assertEqual(paddr, 0x123000)
    self.assertEqual(paddrs, [0x800123000, 0x800124000])
    dev.mm.palloc.assert_called_once_with(0x2000, boot=False, cpu_visible=False)
    dev.pci_dev.bar_info.assert_called_once_with(1)


class TestNVStaleState(unittest.TestCase):
  def test_usb_wpr2_refuses_unproven_hot_reset(self):
    dev = object.__new__(NVDev)
    dev.devfmt, dev.include = "usb:0", MagicMock()
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI = MagicMock()
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read.return_value = 1
    dev.pci_dev = types.SimpleNamespace(wpr_reset_supported=False, reset=MagicMock())

    with self.assertRaisesRegex(RuntimeError, "physically power-cycle"): dev._early_ip_init()

    dev.pci_dev.reset.assert_not_called()


class TestNVLifecycle(unittest.TestCase):
  @staticmethod
  def make_lifecycle_dev(state:NVLifecycleState=NVLifecycleState.READY):
    dev = object.__new__(NVDev)
    dev.lifecycle_state, dev.lifecycle_phase, dev._teardown_completed = state, None, set()
    return dev

  def test_phase_failure_is_not_retried_and_preserves_cause(self):
    dev = object.__new__(NVDev)
    dev.lifecycle_state, dev.lifecycle_phase = NVLifecycleState.SOFTWARE_READY, None
    dev._lifecycle_snapshot = MagicMock(return_value={"wpr2_hi": 0x123})
    failure = ValueError("mailbox 0x0b")
    action = MagicMock(side_effect=failure)

    with self.assertRaises(NVLifecycleError) as raised:
      dev._run_phase(NVLifecyclePhase.BOOTER_LOAD, NVLifecycleState.GSP_RUNNING, action)

    action.assert_called_once_with()
    self.assertIs(raised.exception.__cause__, failure)
    self.assertEqual((raised.exception.phase, raised.exception.state),
                     (NVLifecyclePhase.BOOTER_LOAD, NVLifecycleState.SOFTWARE_READY))
    self.assertEqual(raised.exception.snapshot, {"wpr2_hi": 0x123})
    self.assertEqual((dev.lifecycle_state, dev.lifecycle_phase),
                     (NVLifecycleState.SOFTWARE_READY, NVLifecyclePhase.BOOTER_LOAD))

  def test_full_teardown_has_explicit_order_and_is_idempotent(self):
    events = []
    dev = self.make_lifecycle_dev()
    dev.pci_dev = types.SimpleNamespace(gsp_full_teardown=True)
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI = MagicMock()
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read.side_effect = [1, 0]
    dev.gsp = types.SimpleNamespace(
      rpc_rm_free=lambda obj, client: events.append(("free", obj, client)),
      fini_hw=lambda: events.append(("gsp_suspend",)))
    dev.flcn = object.__new__(NV_FLCN)
    dev.flcn.shutdown_fwsec = lambda: events.append(("fwsec",))
    dev.flcn.shutdown_booter = lambda: events.append(("booter_unload",))
    dev.flcn.set_lifecycle_record_state = lambda state: events.append(("state", state))

    dev.fini(0xc1000000)
    dev.fini(0xc1000000)

    self.assertEqual(events, [("free", 0xc1000000, 0xc1000000), ("state", NVGspLifecycleState.ROOT_RELEASED),
                              ("gsp_suspend",), ("state", NVGspLifecycleState.GSP_SUSPENDED),
                              ("fwsec",), ("state", NVGspLifecycleState.FWSEC_COMPLETE), ("booter_unload",)])
    self.assertEqual((dev.lifecycle_state, dev.lifecycle_phase), (NVLifecycleState.CLOSED, None))

  def test_failed_gsp_suspend_still_runs_secure_teardown(self):
    dev = self.make_lifecycle_dev()
    dev.pci_dev = types.SimpleNamespace(gsp_full_teardown=True)
    dev._lifecycle_snapshot = MagicMock(return_value={"gsp_mailbox0": 0})
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI = MagicMock()
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read.side_effect = [1, 0]
    dev.gsp = types.SimpleNamespace(rpc_rm_free=MagicMock(), fini_hw=MagicMock(side_effect=RuntimeError("not suspended")))
    dev.flcn = object.__new__(NV_FLCN)
    dev.flcn.shutdown_fwsec, dev.flcn.shutdown_booter = MagicMock(), MagicMock()

    with self.assertRaises(NVLifecycleError) as raised: dev.fini(0xc1000000)

    self.assertEqual(raised.exception.phase, NVLifecyclePhase.GSP_SUSPEND)
    self.assertEqual(dev.lifecycle_state, NVLifecycleState.CLOSED)
    dev.flcn.shutdown_fwsec.assert_called_once_with()
    dev.flcn.shutdown_booter.assert_called_once_with()

  def test_wpr_only_partial_boot_skips_gsp_rpc_and_clears_wpr(self):
    dev = self.make_lifecycle_dev(NVLifecycleState.WPR_ACTIVE)
    dev.pci_dev = types.SimpleNamespace(gsp_full_teardown=True)
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI = MagicMock()
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read.side_effect = [1, 0]
    dev.gsp = types.SimpleNamespace(rpc_rm_free=MagicMock(), fini_hw=MagicMock())
    dev.flcn = object.__new__(NV_FLCN)
    dev.flcn.shutdown_fwsec, dev.flcn.shutdown_booter = MagicMock(), MagicMock()

    dev.fini()

    dev.gsp.rpc_rm_free.assert_not_called()
    dev.gsp.fini_hw.assert_not_called()
    dev.flcn.shutdown_fwsec.assert_called_once_with()
    dev.flcn.shutdown_booter.assert_called_once_with()
    self.assertEqual(dev.lifecycle_state, NVLifecycleState.CLOSED)

  def test_secure_teardown_collects_failures_and_attempts_booter(self):
    dev = self.make_lifecycle_dev()
    dev.pci_dev = types.SimpleNamespace(gsp_full_teardown=True)
    dev._lifecycle_snapshot = MagicMock(return_value={"wpr2_hi": 1})
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI = MagicMock()
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read.side_effect = [1, 0]
    dev.gsp = types.SimpleNamespace(rpc_rm_free=MagicMock(), fini_hw=MagicMock(side_effect=RuntimeError("no suspend")))
    dev.flcn = object.__new__(NV_FLCN)
    dev.flcn.shutdown_fwsec = MagicMock(side_effect=RuntimeError("fwsec failed"))
    dev.flcn.shutdown_booter = MagicMock()

    with self.assertRaises(ExceptionGroup) as raised: dev.fini(0xc1000000)

    self.assertEqual([error.phase for error in raised.exception.exceptions],
                     [NVLifecyclePhase.GSP_SUSPEND, NVLifecyclePhase.FWSEC_SHUTDOWN])
    dev.flcn.shutdown_booter.assert_called_once_with()
    self.assertEqual(dev.lifecycle_state, NVLifecycleState.CLOSED)

  def test_final_wpr_read_failure_is_reported(self):
    dev = self.make_lifecycle_dev()
    dev.pci_dev = types.SimpleNamespace(gsp_full_teardown=True)
    dev._lifecycle_snapshot = MagicMock(return_value={"wpr2_hi": "unavailable"})
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI = MagicMock()
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read.side_effect = [1, RuntimeError("WPR read failed")]
    dev.gsp = types.SimpleNamespace(rpc_rm_free=MagicMock(), fini_hw=MagicMock())
    dev.flcn = object.__new__(NV_FLCN)
    dev.flcn.shutdown_fwsec, dev.flcn.shutdown_booter = MagicMock(), MagicMock()

    with self.assertRaises(NVLifecycleError) as raised: dev.fini(0xc1000000)

    self.assertEqual(raised.exception.phase, NVLifecyclePhase.WPR_VERIFY)
    self.assertEqual(dev.lifecycle_state, NVLifecycleState.WPR_CLEARED)

  def test_lifecycle_record_clear_failure_is_reported_after_wpr_clears(self):
    dev = self.make_lifecycle_dev()
    dev.pci_dev = types.SimpleNamespace(gsp_full_teardown=True)
    dev._lifecycle_snapshot = MagicMock(return_value={"wpr2_hi": 0})
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI = MagicMock()
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read.side_effect = [1, 0]
    dev.gsp = types.SimpleNamespace(rpc_rm_free=MagicMock(), fini_hw=MagicMock())
    dev.flcn = object.__new__(NV_FLCN)
    dev.flcn.shutdown_fwsec, dev.flcn.shutdown_booter = MagicMock(), MagicMock()
    dev.flcn.clear_lifecycle_record = MagicMock(side_effect=RuntimeError("record clear failed"))

    with self.assertRaises(NVLifecycleError) as raised: dev.fini(0xc1000000)

    self.assertEqual(raised.exception.phase, NVLifecyclePhase.LIFECYCLE_RECORD_CLEAR)
    self.assertEqual(dev.lifecycle_state, NVLifecycleState.CLOSED)

  def test_full_initialize_prepares_shutdown_before_wpr(self):
    dev = self.make_lifecycle_dev(NVLifecycleState.NEW)
    dev.pci_dev = types.SimpleNamespace(gsp_full_teardown=True)
    dev.flcn = object.__new__(NV_FLCN)
    dev.gsp = types.SimpleNamespace(init_hw=MagicMock())
    dev._early_ip_init, dev._early_mmu_init, dev._init_sw, dev._set_gpu_time = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    dev.flcn.prep_fini, dev.flcn.init_wpr, dev.flcn.boot_gsp = MagicMock(), MagicMock(), MagicMock()
    phases = []
    def run_phase(phase, state, fn):
      phases.append(phase)
      fn()
      dev.lifecycle_state = state
    dev._run_phase = run_phase

    dev.initialize()

    self.assertEqual(phases, [NVLifecyclePhase.PCI_SETUP, NVLifecyclePhase.MMU_SETUP, NVLifecyclePhase.SOFTWARE_SETUP,
                              NVLifecyclePhase.TEARDOWN_PREP, NVLifecyclePhase.WPR_SETUP, NVLifecyclePhase.BOOTER_LOAD,
                              NVLifecyclePhase.GSP_RM_INIT])
    self.assertEqual(dev.lifecycle_state, NVLifecycleState.READY)

  def test_usb_boot_sets_gpu_time_before_booter_load(self):
    dev, events = self.make_lifecycle_dev(), []
    dev.flcn = object.__new__(NV_FLCN)
    dev._set_gpu_time = lambda: events.append("ptimer")
    dev.flcn.boot_gsp = lambda: events.append("booter")
    dev.flcn.set_lifecycle_record_state = lambda state: events.append(state)

    dev._boot_gsp()

    self.assertEqual(events, ["ptimer", "booter", NVGspLifecycleState.GSP_RUNNING])

  def test_gpu_time_writes_high_then_committing_low_word(self):
    dev = object.__new__(NVDev)
    dev.rreg, dev.wreg = MagicMock(return_value=1 << 4), MagicMock()

    with patch("tinygrad.runtime.support.nv.nvdev.time.time_ns", return_value=0x123456789abcdef0): dev._set_gpu_time()

    dev.rreg.assert_called_once_with(0x9430)
    self.assertEqual(dev.wreg.call_args_list, [call(0x9410, 0x12345678), call(0x9400, 0x9abcdef0)])

  def test_gpu_time_rejects_locked_level_zero_writes(self):
    dev = object.__new__(NVDev)
    dev.rreg, dev.wreg = MagicMock(return_value=0), MagicMock()

    with self.assertRaisesRegex(RuntimeError, "PTIMER is not writable"): dev._set_gpu_time()

    dev.wreg.assert_not_called()

  def test_initialize_recovers_stale_state_before_mmu_setup(self):
    dev = self.make_lifecycle_dev(NVLifecycleState.NEW)
    dev.pci_dev = types.SimpleNamespace(gsp_full_teardown=False)
    dev.flcn, dev.gsp = MagicMock(), MagicMock()
    events = []
    dev._early_ip_init = lambda: events.append("pci")
    dev._recover_unclean_state = lambda: events.append("recover")
    dev._early_mmu_init = lambda: events.append("mmu")
    dev._init_sw = lambda: events.append("software")
    dev.flcn.init_hw.side_effect = lambda: events.append("boot")
    dev.gsp.init_hw.side_effect = lambda: events.append("rm")
    dev._unclean_recovery_required = True

    dev.initialize()

    self.assertEqual(events, ["pci", "recover", "mmu", "software", "boot", "rm"])

  def test_initialize_does_not_boot_after_unclean_recovery_failure(self):
    dev = self.make_lifecycle_dev(NVLifecycleState.NEW)
    dev.pci_dev = types.SimpleNamespace(gsp_full_teardown=False)
    dev._lifecycle_snapshot = MagicMock(return_value={"wpr2_hi": 1})
    dev._early_ip_init = lambda: setattr(dev, "_unclean_recovery_required", True)
    dev._recover_unclean_state = MagicMock(side_effect=RuntimeError("queue attachment failed"))
    dev._early_mmu_init, dev._init_sw = MagicMock(), MagicMock()

    with self.assertRaises(NVLifecycleError) as raised: dev.initialize()

    self.assertEqual(raised.exception.phase, NVLifecyclePhase.UNCLEAN_RECOVERY)
    dev._early_mmu_init.assert_not_called()
    dev._init_sw.assert_not_called()

  @staticmethod
  def make_flr_recovery_dev(wpr2_hi:int=0):
    dev = TestNVLifecycle.make_lifecycle_dev(NVLifecycleState.PCI_READY)
    dev.devfmt, dev._unclean_recovery_required = "usb:0", True
    dev.pci_dev = types.SimpleNamespace(gsp_flr_recovery=True, function_level_reset=MagicMock(),
                                         map_bar=MagicMock(return_value="remapped-mmio"))
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI = MagicMock()
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read.return_value = wpr2_hi
    dev.flcn = object.__new__(NV_FLCN)
    dev.flcn.wait_for_reset, dev.flcn.clear_lifecycle_record = MagicMock(), MagicMock()
    return dev

  def test_unclean_recovery_uses_flr_then_starts_from_reset_state(self):
    dev = self.make_flr_recovery_dev()

    dev._recover_unclean_state()

    dev.pci_dev.function_level_reset.assert_called_once_with()
    dev.pci_dev.map_bar.assert_called_once_with(0, fmt='I')
    self.assertEqual(dev.mmio, "remapped-mmio")
    dev.flcn.wait_for_reset.assert_called_once_with()
    dev.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read.assert_called_once_with()
    dev.flcn.clear_lifecycle_record.assert_called_once_with(persisted=True)
    self.assertFalse(dev._unclean_recovery_required)

  def test_unclean_recovery_requires_advertised_flr(self):
    dev = self.make_flr_recovery_dev()
    dev.pci_dev.gsp_flr_recovery = False

    with self.assertRaisesRegex(RuntimeError, "does not support Function Level Reset"):
      dev._recover_unclean_state()

    dev.pci_dev.function_level_reset.assert_not_called()
    dev.pci_dev.map_bar.assert_not_called()
    dev.flcn.wait_for_reset.assert_not_called()

  def test_unclean_recovery_flr_failure_does_not_continue(self):
    dev = self.make_flr_recovery_dev()
    dev.pci_dev.function_level_reset.side_effect = RuntimeError("FLR failed")

    with self.assertRaisesRegex(RuntimeError, "FLR failed"): dev._recover_unclean_state()

    dev.pci_dev.function_level_reset.assert_called_once_with()
    dev.pci_dev.map_bar.assert_not_called()
    dev.flcn.wait_for_reset.assert_not_called()
    dev.flcn.clear_lifecycle_record.assert_not_called()
    self.assertTrue(dev._unclean_recovery_required)

  def test_unclean_recovery_requires_wpr_clear_after_flr(self):
    dev = self.make_flr_recovery_dev(wpr2_hi=1)

    with self.assertRaisesRegex(RuntimeError, "WPR2 remains active after NVIDIA Function Level Reset"):
      dev._recover_unclean_state()

    dev.flcn.wait_for_reset.assert_called_once_with()
    dev.flcn.clear_lifecycle_record.assert_not_called()
    self.assertTrue(dev._unclean_recovery_required)

  def test_unclean_recovery_record_clear_failure_does_not_continue(self):
    dev = self.make_flr_recovery_dev()
    dev.flcn.clear_lifecycle_record.side_effect = RuntimeError("record clear failed")

    with self.assertRaisesRegex(RuntimeError, "record clear failed"): dev._recover_unclean_state()

    self.assertTrue(dev._unclean_recovery_required)


def ifr_vbios_words(version:int=3) -> dict[int, int]:
  words = {0: 0x4947564E, 4: (0x24 << 16) | (version << 8), 0x2000: 0xAA55}
  if version in (1, 2): words[0x28] = 0x2000
  elif version == 3: words.update({8: 0x40, 0x40: 0x100, 0x1100: 0x44524652, 0x1108: 0x2000})
  return words


class TestNVVBIOSLayout(unittest.TestCase):
  def test_standard_pci_rom_starts_at_zero(self):
    self.assertEqual(find_vbios_rom_offset({0: 0xAA55}.get), 0)

  def test_ifr_v1_points_to_pci_rom(self):
    self.assertEqual(find_vbios_rom_offset(ifr_vbios_words(1).get), 0x2000)

  def test_ifr_v3_directory_points_to_pci_rom(self):
    self.assertEqual(find_vbios_rom_offset(ifr_vbios_words(3).get), 0x2000)

  def test_ifr_directory_signature_is_validated(self):
    words = ifr_vbios_words(3)
    words[0x1100] = 0
    with self.assertRaisesRegex(ValueError, "ROM directory signature"):
      find_vbios_rom_offset(words.get)

  def test_ifr_pci_rom_offset_must_be_aligned(self):
    words = ifr_vbios_words(1)
    words[0x28] = 0x2001
    with self.assertRaisesRegex(ValueError, "PCI option ROM offset"):
      find_vbios_rom_offset(words.get)


class TestNVCPUVisibleMemory(unittest.TestCase):
  def test_cpu_visible_and_normal_vram_use_disjoint_allocators(self):
    storage = (ctypes.c_ubyte * (32 << 20))()
    dev = types.SimpleNamespace(
      is_booting=True, smi_dev=False, devfmt="mock", mmu_ver=2,
      vram=MMIOInterface(ctypes.addressof(storage), len(storage)),
      NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE=MagicMock())
    dev.pte_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_PTE'][2])
    dev.pde_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_PDE'][2])
    dev.dual_pde_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_DUAL_PDE'][2])
    NVMemoryManager.va_allocator = TLSFAllocator(1 << 44, base=0x1000000000)
    dev.mm = NVMemoryManager(dev, len(storage), boot_size=2 << 20, pt_t=NVPageTableEntry, va_bits=48,
                             va_shifts=[12, 21, 29, 38, 47], va_base=0,
                             palloc_ranges=[(2 << 20, 2 << 20), (4 << 10, 4 << 10)], reserve_ptable=True,
                             cpu_visible_limit=8 << 20)
    dev.is_booting = False

    low = dev.mm.palloc(2 << 20, zero=False, cpu_visible=True)
    normal = dev.mm.palloc(2 << 20, zero=False)
    self.assertEqual(low, 3 << 20)
    self.assertEqual(normal, 8 << 20)

    dev.mm.pfree(low)
    dev.mm.pfree(normal)
    self.assertEqual(dev.mm.palloc(2 << 20, zero=False, cpu_visible=True), low)
    self.assertEqual(dev.mm.palloc(2 << 20, zero=False), normal)

    with self.assertRaisesRegex(MemoryError, "Can't allocate"):
      dev.mm.palloc(4 << 20, zero=False, cpu_visible=True)

  def test_reserved_gap_is_excluded_from_both_physical_allocators(self):
    storage = (ctypes.c_ubyte * (32 << 20))()
    dev = types.SimpleNamespace(
      is_booting=True, smi_dev=False, devfmt="mock", mmu_ver=2,
      vram=MMIOInterface(ctypes.addressof(storage), len(storage)),
      NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE=MagicMock())
    dev.pte_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_PTE'][2])
    dev.pde_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_PDE'][2])
    dev.dual_pde_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_DUAL_PDE'][2])
    NVMemoryManager.va_allocator = TLSFAllocator(1 << 44, base=0x1000000000)
    dev.mm = NVMemoryManager(dev, len(storage), boot_size=2 << 20, pt_t=NVPageTableEntry, va_bits=48,
                             va_shifts=[12, 21, 29, 38, 47], va_base=0,
                             palloc_ranges=[(2 << 20, 2 << 20), (4 << 10, 4 << 10)], reserve_ptable=True,
                             cpu_visible_limit=8 << 20, pa_start=9 << 20)
    dev.is_booting = False

    self.assertEqual(dev.mm.cpu_visible_pa_allocator.base + dev.mm.cpu_visible_pa_allocator.size, 8 << 20)
    self.assertEqual(dev.mm.pa_allocator.base, 9 << 20)
    self.assertEqual(dev.mm.palloc(2 << 20, zero=False), 9 << 20)


class TestNVPageTableEntry(unittest.TestCase):
  class RecordingEntries:
    def __init__(self): self.writes = []
    def __setitem__(self, key, value): self.writes.append((key, list(value)))

  @staticmethod
  def make_dual_entry(is_page:bool):
    entry = object.__new__(NVPageTableEntry)
    entry.lv, entry.entries = 3, [0] * 512
    entry.entries[24] = int(is_page)
    entry.nvdev = types.SimpleNamespace(
      mm=types.SimpleNamespace(level_cnt=5), mmu_ver=2,
      pte_t=StaticDecoder({'address_sys': 0xE600}),
      dual_pde_t=StaticDecoder({'address_small_sys': 0x214}),
      pde_t=StaticDecoder({}))
    return entry

  def test_dual_level_huge_pte_uses_pte_address(self):
    self.assertEqual(self.make_dual_entry(is_page=True).address(12), 0xE600000)

  def test_dual_level_table_uses_small_table_address(self):
    self.assertEqual(self.make_dual_entry(is_page=False).address(12), 0x214000)

  @staticmethod
  def make_encoding_entry(lv:int):
    entry = object.__new__(NVPageTableEntry)
    entry.lv, entry.entries = lv, TestNVPageTableEntry.RecordingEntries()
    entry.nvdev = types.SimpleNamespace(mm=types.SimpleNamespace(level_cnt=5), mmu_ver=2)
    entry.nvdev.pte_t = NVReg(entry.nvdev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_PTE'][2])
    entry.nvdev.pde_t = NVReg(entry.nvdev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_PDE'][2])
    entry.nvdev.dual_pde_t = NVReg(entry.nvdev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_DUAL_PDE'][2])
    return entry

  def test_set_entries_batches_leaf_ptes(self):
    entry = self.make_encoding_entry(4)
    entry.set_entries(7, [0x2000, 0x5000], uncached=True)
    expected = [entry.nvdev.pte_t.encode(valid=True, address_sys=paddr >> 12, aperture=0, kind=6, vol=True) for paddr in (0x2000, 0x5000)]
    self.assertEqual(entry.entries.writes, [(slice(7, 9), expected)])

  def test_set_entries_batches_dual_level_ptes(self):
    entry = self.make_encoding_entry(3)
    entry.set_entries(7, [0x200000, 0x400000])
    expected = [entry.nvdev.pte_t.encode(valid=True, address_sys=paddr >> 12, aperture=0, kind=6, vol=False)
                for paddr in (0x200000, 0x400000)]
    self.assertEqual(entry.entries.writes, [(slice(14, 18), [expected[0], 0, expected[1], 0])])

  def test_adjacent_huge_mapping_preserves_existing_path(self):
    storage = (ctypes.c_ubyte * (16 << 20))()
    dev = types.SimpleNamespace(
      is_booting=True, smi_dev=False, devfmt="mock", mmu_ver=2,
      vram=MMIOInterface(ctypes.addressof(storage), len(storage)),
      NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE=MagicMock())
    dev.pte_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_PTE'][2])
    dev.pde_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_PDE'][2])
    dev.dual_pde_t = NVReg(dev, None, None, nv_regs.dev_mmu.tu102['NV_MMU_VER2_DUAL_PDE'][2])
    NVMemoryManager.va_allocator = TLSFAllocator(1 << 44, base=0x1000000000)
    dev.mm = NVMemoryManager(dev, len(storage), boot_size=2 << 20, pt_t=NVPageTableEntry, va_bits=48,
                             va_shifts=[12, 21, 29, 38, 47], va_base=0,
                             palloc_ranges=[(2 << 20, 2 << 20), (4 << 10, 4 << 10)], reserve_ptable=True)
    dev.is_booting = False

    first = dev.mm.valloc(2 << 20, contiguous=True)
    dev.mm.valloc(2 << 20, contiguous=True, uncached=True)

    pt = dev.mm.root_page_table
    while not pt.is_page(idx:=(first.va_addr // dev.mm.pte_covers[pt.lv]) % dev.mm.pte_cnt[pt.lv]):
      self.assertTrue(pt.valid(idx))
      pt = dev.mm.pt_t(dev, pt.address(idx), lv=pt.lv + 1)
    self.assertEqual(pt.address(idx), first.paddrs[0][0])


if __name__ == "__main__": unittest.main()
