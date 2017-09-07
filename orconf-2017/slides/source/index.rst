
Bringing up Cycle-accurate models of RISC-V cores
=================================================

Graham Markall

Compiler Engineer, `Embecosm <http://www.embecosm.com/>`_

`graham.markall@embecosm.com <mailto:graham.markall@embecosm.com>`_

Twitter: `@gmarkall <https://twitter.com/gmarkall>`_

Intro
-----

Short story of survey, software, and hardware implementation

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

More detail on each core

Toolchain implementation
------------------------

- Which repos
- Code management strategy
- Library + other customisations
- Verilator / cycle-accurate models

Testing
-------

- GCC test results


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

- How to replicate experiments / results
