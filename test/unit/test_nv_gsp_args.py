import ctypes, types, unittest
from unittest.mock import MagicMock

from tinygrad.runtime.autogen import nv
from tinygrad.runtime.support.nv.ip import NV_GSP


class TestNVGSPArguments(unittest.TestCase):
  def test_rpc_memory_invalidate_waits_for_completion(self):
    gsp = object.__new__(NV_GSP)
    reg = MagicMock()
    reg.read.side_effect = [3, 0]
    gsp.nvdev = types.SimpleNamespace(NV_VIRTUAL_FUNCTION_PRIV_L2_SYSMEM_INVALIDATE=reg)

    gsp.invalidate_rpc_memory()

    reg.write.assert_called_once_with(1)
    self.assertEqual(reg.read.call_count, 2)

  def test_skips_optional_registry_rpc_when_transport_requests_it(self):
    gsp = object.__new__(NV_GSP)
    gsp.nvdev = types.SimpleNamespace(pci_dev=types.SimpleNamespace(skip_gsp_registry=True), chip_name="GA102")
    gsp.init_rm_args = gsp.init_libos_args = gsp.init_wpr_meta = MagicMock()
    gsp.rpc_set_gsp_system_info, gsp.rpc_set_registry_table = MagicMock(), MagicMock()

    gsp.init_sw()

    gsp.rpc_set_gsp_system_info.assert_called_once_with()
    gsp.rpc_set_registry_table.assert_not_called()

  def test_vram_logs_are_described_as_framebuffer_memory(self):
    gsp = object.__new__(NV_GSP)
    stage = MagicMock(return_value=0x828200)
    pci_dev = types.SimpleNamespace(boot_mem_in_vram=True, stage_gsp_libos_args=stage)
    gsp.nvdev = types.SimpleNamespace(pci_dev=pci_dev, _alloc_boot_mem=MagicMock(return_value=(MagicMock(), 0x300000, [0x300000])))
    gsp.rm_args_sysmem = 0x828100

    gsp.init_libos_args()

    self.assertEqual(gsp.libos_args_sysmem, 0x828200)
    args = stage.call_args.args[0]
    struct_size = ctypes.sizeof(nv.LibosMemoryRegionInitArgument)
    regions = [nv.LibosMemoryRegionInitArgument.from_buffer_copy(args, i * struct_size) for i in range(6)]
    self.assertTrue(all(region.loc == nv.LIBOS_MEMORY_REGION_LOC_FB for region in regions[:5]))
    self.assertEqual(regions[5].loc, nv.LIBOS_MEMORY_REGION_LOC_SYSMEM)
    self.assertEqual(regions[5].pa, 0x828100)


if __name__ == "__main__": unittest.main()
