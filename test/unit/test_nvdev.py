import ctypes, types, unittest
from unittest.mock import MagicMock

from tinygrad.runtime.autogen import nv_regs
from tinygrad.runtime.support.hcq import MMIOInterface
from tinygrad.runtime.support.memory import TLSFAllocator
from tinygrad.runtime.support.nv.ip import find_vbios_rom_offset
from tinygrad.runtime.support.nv.nvdev import NVDev, NVMemoryManager, NVPageTableEntry, NVReg


class StaticDecoder:
  def __init__(self, fields): self.fields = fields
  def decode(self, value): return self.fields


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


class TestNVPageTableEntry(unittest.TestCase):
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
