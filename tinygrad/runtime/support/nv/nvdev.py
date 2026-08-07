from __future__ import annotations
import time, functools, enum, array, tinygrad.runtime.autogen.nv_regs
from tinygrad.helpers import getenv, DEBUG, getbits, round_up
from tinygrad.runtime.autogen import pci
from tinygrad.runtime.support.memory import TLSFAllocator, MemoryManager, AddrSpace
from tinygrad.runtime.support.nv.ip import NV_FLCN, NV_FLCN_COT, NV_GSP
from tinygrad.runtime.support.system import PCIDevice
from tinygrad.runtime.support.hcq import MMIOInterface

NV_DEBUG = getenv("NV_DEBUG", 0)

class NVLifecycleState(enum.Enum):
  NEW = "new"
  PCI_READY = "pci_ready"
  MMU_READY = "mmu_ready"
  SOFTWARE_READY = "software_ready"
  SHUTDOWN_READY = "shutdown_ready"
  WPR_ACTIVE = "wpr_active"
  GSP_RUNNING = "gsp_running"
  RM_READY = "rm_ready"
  READY = "ready"
  CLIENT_RELEASED = "client_released"
  GSP_SUSPENDED = "gsp_suspended"
  FWSEC_COMPLETE = "fwsec_complete"
  WPR_CLEARED = "wpr_cleared"
  CLOSED = "closed"

class NVLifecyclePhase(enum.Enum):
  PCI_SETUP = "pci_setup"
  UNCLEAN_RECOVERY = "unclean_recovery"
  MMU_SETUP = "mmu_setup"
  SOFTWARE_SETUP = "software_setup"
  TEARDOWN_PREP = "teardown_prep"
  WPR_SETUP = "wpr_setup"
  BOOTER_LOAD = "booter_load"
  GSP_RM_INIT = "gsp_rm_init"
  RM_CLIENT_RELEASE = "rm_client_release"
  GSP_SUSPEND = "gsp_suspend"
  FWSEC_SHUTDOWN = "fwsec_shutdown"
  BOOTER_UNLOAD = "booter_unload"
  WPR_VERIFY = "wpr_verify"

class NVLifecycleError(RuntimeError):
  def __init__(self, phase:NVLifecyclePhase, state:NVLifecycleState, snapshot:dict[str, int|str], cause:Exception):
    self.phase, self.state, self.snapshot = phase, state, snapshot
    def fmt(value:int|str): return f"{value:#x}" if isinstance(value, int) else value
    details = ", ".join(f"{key}={fmt(value)}" for key, value in snapshot.items())
    super().__init__(f"NVIDIA lifecycle phase {phase.value} failed from {state.value}: {cause}" + (f" ({details})" if details else ""))

class NVReg:
  def __init__(self, nvdev, base, off, fields=None): self.nvdev, self.base, self.off, self.fields = nvdev, base, off, fields

  def __getitem__(self, idx:int): return NVReg(self.nvdev, self.base, self.off(idx), fields=self.fields)

  def add_field(self, name:str, start:int, end:int): self.fields[name] = (start, end)
  def with_base(self, base:int): return NVReg(self.nvdev, base + self.base, self.off, self.fields)

  def read(self): return self.nvdev.rreg(self.base + self.off)
  def read_bitfields(self) -> dict[str, int]: return self.decode(self.read())

  def write(self, _ini_val:int=0, **kwargs): self.nvdev.wreg(self.base + self.off, _ini_val | self.encode(**kwargs))

  def update(self, **kwargs): self.write(self.read() & ~self.mask(*kwargs.keys()), **kwargs)

  def mask(self, *names):
    return functools.reduce(int.__or__, ((((1 << (self.fields[nm][1]-self.fields[nm][0] + 1)) - 1) << self.fields[nm][0]) for nm in names), 0)

  def encode(self, **kwargs) -> int: return functools.reduce(int.__or__, (value << self.fields[name][0] for name,value in kwargs.items()), 0)
  def decode(self, val: int) -> dict: return {name:getbits(val, start, end) for name,(start,end) in self.fields.items()}

class NVPageTableEntry:
  def __init__(self, nvdev, paddr, lv): self.nvdev, self.paddr, self.lv, self.entries = nvdev, paddr, lv, nvdev.vram.view(paddr, 0x1000, fmt='Q')

  def _is_dual_pde(self) -> bool: return self.lv == self.nvdev.mm.level_cnt - 2

  def _encode_entry(self, paddr:int, table=False, uncached=False, aspace=AddrSpace.PHYS, snooped=False, frag=0, valid=True) -> int:
    if not table:
      aperture = (2 if snooped else 3) if aspace is AddrSpace.SYS else 0
      return self.nvdev.pte_t.encode(valid=valid, address_sys=paddr >> 12, aperture=aperture, kind=6,
        **({'pcf': int(uncached)} if self.nvdev.mmu_ver == 3 else {'vol': uncached}))
    pde = self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t
    small, sys = ("_small" if self._is_dual_pde() else ""), "" if self.nvdev.mmu_ver == 3 else "_sys"
    return pde.encode(is_pte=False, **{f'aperture{small}': 1 if valid else 0, f'address{small}{sys}': paddr >> 12},
      **({f'pcf{small}': 0b10} if self.nvdev.mmu_ver == 3 else {'no_ats': 1}))

  def set_entries(self, entry_id:int, paddrs:list[int], table=False, uncached=False, aspace=AddrSpace.PHYS, snooped=False, frag=0, valid=True):
    if not paddrs: return
    encoded = [self._encode_entry(paddr, table, uncached, aspace, snooped, frag, valid) for paddr in paddrs]
    words = [word for x in encoded for word in (x & 0xffffffffffffffff, x >> 64)] if self._is_dual_pde() else encoded
    start = entry_id * (2 if self._is_dual_pde() else 1)
    self.entries[start:start+len(words)] = array.array('Q', words)

  def set_entry(self, entry_id:int, paddr:int, table=False, uncached=False, aspace=AddrSpace.PHYS, snooped=False, frag=0, valid=True):
    self.set_entries(entry_id, [paddr], table, uncached, aspace, snooped, frag, valid)

  def entry(self, entry_id:int) -> int:
    return (self.entries[2*entry_id+1]<<64) | self.entries[2*entry_id] if self._is_dual_pde() else self.entries[entry_id]

  def read_fields(self, entry_id:int) -> dict:
    if self.is_page(entry_id): return self.nvdev.pte_t.decode(self.entry(entry_id))
    return (self.nvdev.dual_pde_t if self._is_dual_pde() else self.nvdev.pde_t).decode(self.entry(entry_id))

  def is_page(self, entry_id) -> bool: return (self.entry(entry_id) & 1 == 1) if self.lv < self.nvdev.mm.level_cnt - 1 else True
  def supports_huge_page(self, paddr:int): return self.lv >= self.nvdev.mm.level_cnt - 3 and paddr % self.nvdev.mm.pte_covers[self.lv] == 0

  def valid(self, entry_id):
    if self.is_page(entry_id): return self.read_fields(entry_id)['valid']
    return self.read_fields(entry_id)['aperture_small' if self._is_dual_pde() else 'aperture'] != 0

  def address(self, entry_id:int) -> int:
    if self.is_page(entry_id): return self.read_fields(entry_id)['address_sys'] << 12
    small, sys = ("_small" if self._is_dual_pde() else ""), "_sys" if self.nvdev.mmu_ver == 2 or self.lv == self.nvdev.mm.level_cnt - 1 else ""
    return self.read_fields(entry_id)[f'address{small}{sys}'] << 12

class NVMemoryManager(MemoryManager):
  va_allocator = TLSFAllocator((1 << 44), base=0x1000000000) # global for all devices.

  def on_range_mapped(self): self.dev.NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE.write((1 << 0) | (1 << 1) | (1 << 6) | (1 << 31))

class NVDev:
  def __init__(self, pci_dev:PCIDevice):
    self.pci_dev, self.devfmt, self.mmio = pci_dev, pci_dev.pcibus, pci_dev.map_bar(0, fmt='I')
    self.lifecycle_state:NVLifecycleState = NVLifecycleState.NEW
    self.lifecycle_phase:NVLifecyclePhase|None = None
    self._teardown_completed:set[NVLifecyclePhase] = set()

    self.smi_dev, self.is_booting, self.is_err_state = False, True, False
    self.initialize()

  def initialize(self):
    if self.lifecycle_state is not NVLifecycleState.NEW:
      raise RuntimeError(f"NVIDIA device initialization cannot start from {self.lifecycle_state.value}")
    self._run_phase(NVLifecyclePhase.PCI_SETUP, NVLifecycleState.PCI_READY, self._early_ip_init)
    if getattr(self, "_unclean_recovery_required", False):
      self._run_phase(NVLifecyclePhase.UNCLEAN_RECOVERY, NVLifecycleState.PCI_READY, self._recover_unclean_state)
    self._run_phase(NVLifecyclePhase.MMU_SETUP, NVLifecycleState.MMU_READY, self._early_mmu_init)

    # No booting state, gsp client is reinited every run.
    self.is_booting = False

    self._run_phase(NVLifecyclePhase.SOFTWARE_SETUP, NVLifecycleState.SOFTWARE_READY, self._init_sw)
    if getattr(self.pci_dev, "gsp_full_teardown", False) and isinstance(self.flcn, NV_FLCN):
      self._run_phase(NVLifecyclePhase.TEARDOWN_PREP, NVLifecycleState.SHUTDOWN_READY, self.flcn.prep_fini)
      self._run_phase(NVLifecyclePhase.WPR_SETUP, NVLifecycleState.WPR_ACTIVE, self.flcn.init_wpr)
      self._run_phase(NVLifecyclePhase.BOOTER_LOAD, NVLifecycleState.GSP_RUNNING, self._boot_gsp)
    else: self._run_phase(NVLifecyclePhase.BOOTER_LOAD, NVLifecycleState.GSP_RUNNING, self.flcn.init_hw)
    self._run_phase(NVLifecyclePhase.GSP_RM_INIT, NVLifecycleState.RM_READY, self.gsp.init_hw)
    self.lifecycle_state = NVLifecycleState.READY

  def _init_sw(self):
    for ip in [self.flcn, self.gsp]: ip.init_sw()

  def _boot_gsp(self):
    assert isinstance(self.flcn, NV_FLCN)
    self._set_gpu_time()
    self.flcn.boot_gsp()

  def _set_gpu_time(self):
    if not self.rreg(0x9430) & (1 << 4): raise RuntimeError("NVIDIA PTIMER is not writable from privilege level 0")
    ns = time.time_ns()
    self.wreg(0x9410, ns >> 32)
    self.wreg(0x9400, ns & 0xffffffff) # TIME_0 commits both halves.

  def _lifecycle_snapshot(self) -> dict[str, int|str]:
    snapshot:dict[str, int|str] = {"state": self.lifecycle_state.value}
    def record(name, read):
      try: snapshot[name] = read()
      except Exception: snapshot[name] = "unavailable"

    if hasattr(self, "NV_PFB_PRI_MMU_WPR2_ADDR_HI"): record("wpr2_hi", self.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read)
    if hasattr(self, "NV_PBUS_VBIOS_SCRATCH"): record("fwsec_scratch15", lambda: self.NV_PBUS_VBIOS_SCRATCH[0x15].read())
    if hasattr(self, "flcn") and hasattr(self.flcn, "falcon"):
      record("gsp_mailbox0", lambda: self.reg("NV_PFALCON_FALCON_MAILBOX0").with_base(self.flcn.falcon).read())
    if hasattr(self, "flcn") and hasattr(self.flcn, "sec2"):
      record("sec2_mailbox0", lambda: self.reg("NV_PFALCON_FALCON_MAILBOX0").with_base(self.flcn.sec2).read())
    record("pci_command", lambda: self.pci_dev.read_config(pci.PCI_COMMAND, 2))
    return snapshot

  def _recover_unclean_state(self):
    if not isinstance(flcn:=self.flcn, NV_FLCN):
      raise RuntimeError("Unclean NVIDIA recovery is only implemented for pre-Hopper Falcon GSP boot")
    flr = getattr(self.pci_dev, "function_level_reset", None)
    if not getattr(self.pci_dev, "gsp_flr_recovery", False) or not callable(flr):
      raise RuntimeError("NVIDIA USB device does not support Function Level Reset; physically power-cycle the GPU and USB bridge")
    flr()
    self.mmio = self.pci_dev.map_bar(0, fmt='I')
    flcn.wait_for_reset()
    if self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read() != 0:
      raise RuntimeError("WPR2 remains active after NVIDIA Function Level Reset")
    self._unclean_recovery_required = False

  def _run_phase(self, phase:NVLifecyclePhase, success_state:NVLifecycleState, fn):
    previous_state, self.lifecycle_phase = self.lifecycle_state, phase
    try: ret = fn()
    except Exception as exc:
      if isinstance(exc, NVLifecycleError): raise
      raise NVLifecycleError(phase, previous_state, self._lifecycle_snapshot(), exc) from exc
    self.lifecycle_state, self.lifecycle_phase = success_state, None
    return ret

  def _run_teardown_phase(self, phase:NVLifecyclePhase, success_state:NVLifecycleState, fn, errors:list[Exception]):
    if phase in self._teardown_completed: return
    try: self._run_phase(phase, success_state, fn)
    except Exception as exc: errors.append(exc)
    else: self._teardown_completed.add(phase)

  @staticmethod
  def _raise_lifecycle_errors(message:str, errors:list[Exception]):
    if len(errors) == 1: raise errors[0]
    if errors: raise ExceptionGroup(message, errors)

  def fini(self, client_root:int|None=None):
    if self.lifecycle_state is NVLifecycleState.CLOSED: return

    if not getattr(self.pci_dev, "gsp_full_teardown", False):
      if self.lifecycle_state in (NVLifecycleState.GSP_RUNNING, NVLifecycleState.RM_READY, NVLifecycleState.READY):
        self._run_phase(NVLifecyclePhase.GSP_SUSPEND, NVLifecycleState.CLOSED, self.gsp.fini_hw)
      else: self.lifecycle_state, self.lifecycle_phase = NVLifecycleState.CLOSED, None
      return
    flcn = getattr(self, "flcn", None)
    if flcn is not None and not isinstance(flcn, NV_FLCN):
      raise RuntimeError("Full NVIDIA USB teardown is only implemented for pre-Hopper Falcon GSP boot")

    errors:list[Exception] = []
    gsp_started = self.lifecycle_state in (NVLifecycleState.GSP_RUNNING, NVLifecycleState.RM_READY, NVLifecycleState.READY,
      NVLifecycleState.CLIENT_RELEASED, NVLifecycleState.GSP_SUSPENDED)
    rm_client_ready = self.lifecycle_state is NVLifecycleState.READY

    if client_root is not None and rm_client_ready:
      self._run_teardown_phase(NVLifecyclePhase.RM_CLIENT_RELEASE, NVLifecycleState.CLIENT_RELEASED,
                               lambda: self.gsp.rpc_rm_free(client_root, client=client_root), errors)
    if gsp_started:
      self._run_teardown_phase(NVLifecyclePhase.GSP_SUSPEND, NVLifecycleState.GSP_SUSPENDED, self.gsp.fini_hw, errors)

    wpr_active = False
    if hasattr(self, "NV_PFB_PRI_MMU_WPR2_ADDR_HI"):
      try: wpr_active = self.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read() != 0
      except Exception as exc:
        errors.append(NVLifecycleError(NVLifecyclePhase.FWSEC_SHUTDOWN, self.lifecycle_state, self._lifecycle_snapshot(), exc))
    if wpr_active:
      if not isinstance(flcn, NV_FLCN):
        errors.append(NVLifecycleError(NVLifecyclePhase.FWSEC_SHUTDOWN, self.lifecycle_state, self._lifecycle_snapshot(),
                                       RuntimeError("WPR2 is active but Falcon teardown was not initialized")))
      else:
        self._run_teardown_phase(NVLifecyclePhase.FWSEC_SHUTDOWN, NVLifecycleState.FWSEC_COMPLETE, flcn.shutdown_fwsec, errors)
        self._run_teardown_phase(NVLifecyclePhase.BOOTER_UNLOAD, NVLifecycleState.WPR_CLEARED, flcn.shutdown_booter, errors)

    try: wpr_active = hasattr(self, "NV_PFB_PRI_MMU_WPR2_ADDR_HI") and self.NV_PFB_PRI_MMU_WPR2_ADDR_HI.read() != 0
    except Exception as exc:
      wpr_active = True
      errors.append(NVLifecycleError(NVLifecyclePhase.WPR_VERIFY, self.lifecycle_state, self._lifecycle_snapshot(), exc))
    if wpr_active and not any(isinstance(error, NVLifecycleError) and error.phase is NVLifecyclePhase.WPR_VERIFY for error in errors):
      errors.append(NVLifecycleError(NVLifecyclePhase.WPR_VERIFY, self.lifecycle_state, self._lifecycle_snapshot(),
                                     RuntimeError("WPR2 remains active after NVIDIA teardown")))
    if not wpr_active: self.lifecycle_state, self.lifecycle_phase = NVLifecycleState.CLOSED, None
    self._raise_lifecycle_errors("NVIDIA lifecycle teardown failed", errors)

  def reg(self, reg:str) -> NVReg: return self.__dict__[reg]
  def wreg(self, addr:int, value:int):
    self.mmio[addr // 4] = value
    if NV_DEBUG >= 4: print(f"wreg: {hex(addr)} = {hex(value)}")
  def rreg(self, addr:int) -> int: return self.mmio[addr // 4]

  def _early_ip_init(self):
    self.reg_names:set[str] = set()
    self.reg_offsets:dict[str, tuple[int, int]] = {}

    self.include("nv_ref", "")
    self.include("dev_fb", "tu102")
    self.include("dev_gc6_island", "ga102")

    self._unclean_recovery_required = False
    if self.reg("NV_PFB_PRI_MMU_WPR2_ADDR_HI").read() != 0:
      if getattr(self.pci_dev, "gsp_flr_recovery", False): self._unclean_recovery_required = True
      elif not getattr(self.pci_dev, "wpr_reset_supported", True):
        raise RuntimeError("NVIDIA USB device has active WPR2 state without a supported reset; "
                           "physically power-cycle the GPU and USB bridge")
      else:
        self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) & ~pci.PCI_COMMAND_MASTER, 2)
        if DEBUG >= 2: print(f"nv {self.devfmt}: WPR2 is up. Issuing a full reset.", flush=True)
        self.pci_dev.reset()
        time.sleep(0.1) # wait until device can respond again

    self.pci_dev.write_config_flush(pci.PCI_COMMAND, self.pci_dev.read_config(pci.PCI_COMMAND, 2) | pci.PCI_COMMAND_MASTER, 2)
    self.chip_id = self.reg("NV_PMC_BOOT_0").read()
    self.chip_details = self.reg("NV_PMC_BOOT_42").read_bitfields()
    self.chip_name = {0x17: "GA1", 0x19: "AD1", 0x1b: "GB2"}[self.chip_details['architecture']] + f"{self.chip_details['implementation']:02d}"
    self.fw_name = {"GB2": "gb202", "AD1": "ad102", "GA1": "ga102"}[self.chip_name[:3]]
    self.mmu_ver, self.fmc_boot = (3, True) if self.chip_details['architecture'] >= 0x1a else (2, False)

    self.flcn:NV_FLCN|NV_FLCN_COT = NV_FLCN_COT(self) if self.fmc_boot else NV_FLCN(self)
    self.gsp:NV_GSP = NV_GSP(self)

    if not self._unclean_recovery_required: self.flcn.wait_for_reset()

  def _early_mmu_init(self):
    self.include("dev_vm", "tu102")

    # MMU Init
    self.include("dev_mmu", "gh100" if self.mmu_ver == 3 else "tu102")
    self.pte_t, self.pde_t, self.dual_pde_t = [self.__dict__[name] for name in [f'NV_MMU_VER{self.mmu_ver}_PTE', f'NV_MMU_VER{self.mmu_ver}_PDE',
                                                                                f'NV_MMU_VER{self.mmu_ver}_DUAL_PDE']]

    self.vram_size = self.reg("NV_PGC6_AON_SECURE_SCRATCH_GROUP_42").read() << 20

    self.vram, self.mmio = self.pci_dev.map_bar(1), self.pci_dev.map_bar(0, fmt='I')
    self.large_bar = self.vram.nbytes >= self.vram_size

    # UVM depth   HW level                            VA bits
    # 0           PDE4                                56:56 (hopper+)
    # 1           PDE3                                55:47
    # 2           PDE2                                46:38
    # 3           PDE1 (or 512M PTE)                  37:29
    # 4           PDE0 (dual 64k/4k PDE, or 2M PTE)   28:21
    # 5           PTE_64K / PTE_4K                    20:16 / 20:12
    bits, shifts = (56, [12, 21, 29, 38, 47, 56]) if self.mmu_ver == 3 else (48, [12, 21, 29, 38, 47])

    # tail vram reserved for falcon structs
    cpu_visible_limit = self.pci_dev.bar_info(1)[1] if getattr(self.pci_dev, "boot_mem_in_vram", False) else None
    self.mm = NVMemoryManager(self, self.vram_size - (64 << 20), boot_size=(2 << 20), pt_t=NVPageTableEntry, va_bits=bits, va_shifts=shifts,
      va_base=0, palloc_ranges=[(x, x) for x in [512 << 20, 2 << 20, 4 << 10]], reserve_ptable=not self.large_bar,
      cpu_visible_limit=cpu_visible_limit)

  def _alloc_boot_mem(self, size:int, data:bytes|None=None, contiguous:bool=False, sysmem:bool|None=None) -> tuple[MMIOInterface,int|None,list[int]]:
    sz = round_up(size, 0x1000)
    if sysmem is True or (sysmem is None and not self.large_bar and not getattr(self.pci_dev, "boot_mem_in_vram", False)):
      view, sysaddr = self.pci_dev.alloc_sysmem(size, 0, contiguous=contiguous)
      paddr = None
    else:
      cpu_visible = getattr(self.pci_dev, "boot_mem_in_vram", False)
      paddr = self.mm.palloc(sz, boot=False, cpu_visible=cpu_visible)
      view = self.vram.view(paddr, sz)
      base = paddr if cpu_visible else self.pci_dev.bar_info(1)[0] + paddr
      sysaddr = [base + i * 0x1000 for i in range(sz // 0x1000)]
    if data is not None: view[:size] = data
    return view, paddr, sysaddr

  def include(self, name:str, arch:str):
    for k,v in getattr(getattr(tinygrad.runtime.autogen.nv_regs, name), arch or 'regs').items():
      self.__dict__[k] = NVReg(self, *v) if isinstance(v, tuple) else v
