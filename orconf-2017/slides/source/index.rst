
Bringing up Cycle-accurate models of RISC-V cores
=================================================

Graham Markall

Compiler Engineer, `Embecosm <http://www.embecosm.com/>`_

`graham.markall@embecosm.com <mailto:graham.markall@embecosm.com>`_

Twitter: `@gmarkall <https://twitter.com/gmarkall>`_

A Short Story
-------------

- Plot: Going from project idea fully-working system
- Focus here on RISC-V ecosystem and experience
- Survey
- Software toolchain
- Simulation / Testing
- Benchmarking
- Conclusion: **success**

About Me/Embecosm
-----------------

- GCC / GNU Toolchain engineer
- RISC-V / SECURE Projects

.. image:: /_static/about-embecosm.png

Requirements Overview
---------------------

- RV32 / bare metal
- Easily extensible
- Relatively small
- Relatively fast
- Open source


Initial Survey (1-11)
---------------------

- ASTC Systems
- Bluespec
- Clarvi - Simon Moore / Rob Mullins, Cambridge University
- Codasip
- f32c
- lowRISC - lowRISC not-for-profit / Cambridge University
- mriscv - OnchipUIS / Ckristian Duran
- Orca - VectorBlox
- Phalanx - Jan Gray / GRVI
- PicoRV32 - Clifford Wolf
- RI5CY - PuLP Platform / ETHZ


Initial Survey (12-20)
----------------------

- River - GNSS Sensor
- Rocket - Free Chips Project / UCB-Bar
- Shakti - IIT Madras
- Sodor - UC Berkeley / Christopher Celio
- Tom Thumb - Maik Merten
- TurboRav - Sebastian Boe
- uRV - CERN / Tomasz Wlostowski
- Yarvi - Tommy Thorn
- Z-Scale - UC Berkeley / Yunsup Lee, Albert Ou, Albert Magyar


Interesting Cores
-----------------

- PicoRV32
- RI5CY
- Rocket Chip Generator


PicoRV32 - Clifford Wolf
------------------------

- `https://github.com/cliffordwolf/picorv32 <https://github.com/cliffordwolf/picorv32>`_

================= ===================================
Requirement       Assessment
================= ===================================
RV32              Yes (RV32IMC)
Easily extensible Yes, + PCPI
Relatively small  Yes (750-1K LUTs / 397 PLBs)
Relatively fast   Yes (400-700MHz on Xilinx 7-series)
Open source       Yes
================= ===================================


RI5CY - PuLP Platform
---------------------

- `https://github.com/pulp-platform/riscv <https://github.com/pulp-platform/riscv>`_
- `http://www.pulp-platform.org/wp-content/uploads/2017/08/ri5cy_user_manual.pdf <http://www.pulp-platform.org/wp-content/uploads/2017/08/ri5cy_user_manual.pdf>`_

================= ===================================
Requirement       Assessment
================= ===================================
RV32              Yes (RV32IMC + F + Xpulp)
Easily extensible Yes
Relatively small  "Yes"
Relatively fast   ? (50-75MHz on Zynq)
Open source       Yes
================= ===================================


Rocket Chip Generator
---------------------

- `https://github.com/freechipsproject/rocket-chip <https://github.com/freechipsproject/rocket-chip>`_

================= ===================================
Requirement       Assessment
================= ===================================
RV32              Yes (RV32/64 Various)
Easily extensible No (Scala + CHISEL)
Relatively small  Possibly not
Relatively fast   ?
Open source       Yes
================= ===================================


Toolchain implementation
------------------------

GNU Toolchain for bare metal. Upstream snapshots:

- Binutils
- GCC
- GDB
- Newlib

Other repos:

- PicoRV32 + Verilator model + testbench
- RI5CY + Verilator model + testbench
- GDBServer incorporating models

Toolchain customisations
------------------------

Bare metal, so:

- Set SP in :code:`_start`. 4 byte aligned, then 16 bytes
- I/O - GDBServer implements hosted I/O
- Syscall implementation in GDBServer
- Add interrupt vector table for RI5CY

Other Points
------------

- Still link libgloss
- riscv32-unknown-elf naming
- PicoRV32 quite straightforward
- RI5CY always starts at boot address
- Not all SystemVerilog supported by Verilator
- RI5CY memory interface - trial and error

Testing
-------

- RISC-V ISA Test suite
- GCC test suite

PicoRV32 GCC Testsuite
----------------------

========================== =====
Outcome                    Count
========================== =====
Expected passes            86143
Unexpected failures        530
Unexpected successes       4
Expected failures          147
Unresolved testcases       124
Unsupported tests          2540
========================== =====

- Fails: Unimplemented I/O
- Unresolved: timeout too short

RI5CY GCC Testsuite
-------------------

========================== =====
Outcome                    Count
========================== =====
Expected passes            86842
Unexpected failures        27
Unexpected successes       4
Expected failures          147
Unresolved testcases       189
Unsupported tests          2540
========================== =====

- Fails: ctor/dtor, fence, RAM, libunwind, hosted env, upstream
- Unresolved: timeout too short. Down to 7 with more time.

BEEBS: Bristol / Embecosm Embedded Benchmark Suite
--------------------------------------------------

- 81 benchmarks from WCET, MiBench, DSPStone
- Chosen to show the energy consumption of embedded devices
- ARM (STM32), AVR (ATMega328 / 256), X86
- `Machine-Guided Energy-Efficient Compilation (MAGEEC) <http://mageec.org/>`_
- Small, no I/O needed (start / stop triggers)


Discarded Benchmarks
--------------------

- Mostly timeouts
- 1 or 2 self-check issues
- Self-check issues consistent between PicoRV32 and RI5CY

========== ===============
crc32      rijndael
cubic      sglib-arraysort
fdct       sqrt
matmult    trio
nbody      whetstone
nettle-md5 wikisort
========== ===============


Cycle count ratio
-----------------

- See accompanying file, `benchmark.ods <https://github.com/gmarkall/tutorials/blob/master/orconf-2017/benchmark_data.ods?raw=true>`_
- Ratio of PicoRV32 cycle count : RI5CY cycle count
- Mean: 4

.. image:: /_static/corecomparison.png


Conclusions
-----------

- Moving forward with RI5CY core
- Cycle accurate models + toolchain up and running
- RISC-V Ecosystem provided for our needs with low effort
- Building models + toolchain, replicating results:
- `https://github.com/embecosm/riscv-toolchain/tree/orconf <https://github.com/embecosm/riscv-toolchain/tree/orconf>`_
- See README.md
- `Embecosm is hiring <http://www.embecosm.com/2017/08/29/compiler-engineers-wanted-2/>`_!
