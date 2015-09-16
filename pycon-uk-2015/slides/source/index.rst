
.. Accelerating Scientific Code with Numba slides file, created by
   hieroglyph-quickstart on Wed Jun 10 17:01:26 2015.


Accelerating Scientific Code with Numba
=======================================

Graham Markall

Software Engineer, Continuum Analytics

graham.markall@continuum.io

@gmarkall


Tutorial setup
--------------

- Get tutorial material:
  https://gmarkall.github.io/tutorials/pycon-uk-2015/tutorial.zip
- Get Miniconda: https://conda.pydata.org/miniconda.html
- Create environment for exercises:

.. code-block:: bash

   $ conda env create -f=environment.yml


Hello! (about me)
-----------------

- SW Engineer at Continuum Analytics
- Interests in compilers, software performance optimisation, parallel programming
- Background in Python libraries for HPC (PyOP2, Firedrake)
- Numba developer, and Numba user for customer projects


Overview
--------

1. Overview of Numba / Numba basics
2. Understanding Numba / Numba internals
3. Using Numba in your code

Format:

   - Presentation
   - Tutorial exercises (``notebooks`` folder)


What is Numba? (1)
------------------

A tool that makes Python code go faster by specialising and compiling it.

* Mainly focused on array-oriented and numerical code
* Heavily object-oriented, dynamic code not the target use case
* Alternative to using native code (e.g. C-API or CFFI) with C, Fortran, or
  Cython.
* Keeping all code as Python code:

  - Allows focus on algorithmic development
  - Minimise development time

What is Numba? (2)
------------------

* It's *opt-in*: Numba only compiles the functions you specify
* Trade off: relaxing the semantics of Python code in return for performance.


Implementation overview
-----------------------

* JIT compiler for Python based on LLVM
* C/C++ Compiler not required
* Targets CPUs and CUDA GPUs
* CPython 2.6, 2.7, 3.3, 3.4, 3.5

  - Runs side-by-side with Numpy, Scipy, etc ecosystem

* BSD licensed
* Linux / OS X / Windows


Compiler Overview
-----------------

.. image:: /_static/compiler.png


Numba Overview
--------------

.. image:: /_static/numbacompiler.png


Mandelbrot, 20 iterations
-------------------------

============================= =====
CPython                       1x
Numpy array-wide operations   13x
Numba (CPU)                   120x
Numba (NVidia Tesla K20c)     2100x
============================= =====

.. image:: /_static/mandel.png


Mandelbrot function
-------------------

.. code-block:: python

    from numba import jit

    @jit
    def mandel(x, y, max_iters):
        c = complex(x,y)
        z = 0j
        for i in range(max_iters):
            z = z*z + c
            if z.real * z.real + z.imag * z.imag >= 4:
                return 255 * i // max_iters

        return 255


Supported Python Syntax
-----------------------

Inside functions decorated with `@jit`:

* if / else / for / while / break / continue
* raising exceptions
* calling other compiled functions (Numba, Ctypes, CFFI)
* generators!


Unsupported Python Syntax
-------------------------

Also inside functions decorated with `@jit`:

* try / except / finally
* with
* (list, set, dict) comprehensions
* yield from

Classes cannot be decorated with `@jit`.


Supported Python Features
-------------------------

* Types:

    - int, bool, float, complex
    - tuple, None
    - bytes, bytearray, memoryview (and other buffer-like objects)

* Built-in functions:

    - abs, enumerate, len, min, max, print, range, round, zip


Supported Python modules
------------------------

* Standard library:

    - cmath, math, random, ctypes...

* Third-party:

    - cffi, numpy

Comprehensive list: http://numba.pydata.org/numba-doc/0.19.1/reference/pysupported.html


Supported Numpy features
------------------------

* All kinds of arrays: scalar and structured type

    - except when containing Python objects

* Allocation, iterating, indexing, slicing
* Reductions: argmax(), max(), prod() etc.
* Scalar types and values (including datetime64 and timedelta64)
* Array expressions, but no broadcasting
* See reference manual: http://numba.pydata.org/numba-doc/0.19.1/reference/numpysupported.html


Tutorial exercise 1.1
=====================

`The jit decorator`

- Get tutorial material:
  https://gmarkall.github.io/tutorials/pycon-uk-2015/tutorial.zip
- Get Miniconda: https://conda.pydata.org/miniconda.html
- Create environment for exercises:

.. code-block:: bash

   $ conda env create -f=environment.yml



Writing Ufuncs
--------------

* Numpy Universal Function: operates on numpy arrays in an element-by-element fashion
* Supports array broadcasting, casting, reduction, accumulation, etc.

.. code:: python

    @vectorize
    def rel_diff(x, y):
        return 2 * (x - y) / (x + y)

Call:

.. code:: python

    a = np.arange(1000, dtype = float32)
    b = a * 2 + 1
    rel_diff(a, b)


Tutorial exercise 1.2
=====================

`The vectorize decorator`


Generalized Ufuncs
------------------

* Operate on an arbitrary number of elements. Example:

.. code:: python

    @guvectorize([(int64[:], int64[:], int64[:])], '(n),()->(n)')
    def g(x, y, res):
        for i in range(x.shape[0]):
            res[i] = x[i] + y[0]

* No return value: output is passed in
* Input and output layouts: ``(n),()->(n)``
* Before ``->``: Inputs, not allocated. After: outputs, allocated
* Also allows in-place modification


Layout examples
---------------

Matrix-vector products:

.. code:: python

    @guvectorize([(float64[:, :], float64[:], float64[:])],
                  '(m,n),(n)->(m)'

Fixed outputs (e.g. max and min):

.. code:: python

    @guvectorize([(float64[:], float64[:], float64[:])],
                  '(n)->(),()')


Tutorial exercise 1.3
=====================

`The guvectorize decorator`


Understanding Numba / Numba Internals
=====================================

* Numba call performance: dispatch process
* Numba compilation pipeline, and typing
* Nopython mode, object mode, and loop lifting


Dispatch overhead
-----------------

.. code-block:: python

    @jit
    def add(a, b):
        return a + b

    def add_python(a, b):
        return a + b

.. code-block:: python

    >>> %timeit add(1, 2)
    10000000 loops, best of 3: 163 ns per loop

    >>> %timeit add_python(1, 2)
    10000000 loops, best of 3: 85.3 ns per loop


Dispatch process
----------------

Calling a ``@jit`` function:

1. Lookup types of arguments
2. Do any compiled versions match the types of these arguments?

  a. Yes: retrieve the compiled code from the cache
  b. No: compile a new specialisation

3. Marshal arguments to native code
4. Call the native code function
5. Marshal the native return value to a Python value


Compilation pipeline
--------------------

.. image:: /_static/archi2.png
    :width: 400


Type Inference
--------------

* Native code is statically typed, Python is not
* Numba has to determine types by propagating type information
* Uses: mappings of input to output types, and the data flow graph

.. code-block:: python

    def f(a, b):   # a:= float32, b:= float32
        c = a + b  # c:= float32
        return c   # return := float32


Type Unification
----------------

Example typing 1:

.. code-block:: python

    def select(a, b, c):  # a := float32, b := float32, c := bool
        if c:
            ret = a       # ret := float32
        else:
            ret = b       # ret := float32
        return ret       # return := {float32, float32}
                          #           => float32


Type Unification
----------------

Example typing 2:

.. code-block:: python

    def select(a, b, c):  # a := tuple(int32, int32), b := float32,
                          # c := bool
        if c:
            ret = a       # ret := tuple(int32, int32)
        else:
            ret = b       # ret := float32
        return ret       # return := {tuple(int32, int32), float32}
                          #           => XXX

Unification error
-----------------

.. code-block:: none

    numba.typeinfer.TypingError: Failed at nopython (nopython frontend)
    Var 'q1mq0t' unified to object:
        q1mq0t := {array(float64, 1d, C), float64}


.. code-block:: python

    if cond:
        q1mq0t = 6.0
    else:
        q1mq0t = np.zeros(10)

* Treating a variable as an array in one place and a scalar in another


Inspecting pipeline stage output
--------------------------------

* `inspect_types()`
* `inspect_llvm()`
* `inspect_asm()`
* Environment variable ``NUMBA_DEBUG=1``


Tutorial Exercise 2.1
=====================

`Inspection`


Interpreting Type Errors
------------------------

.. code-block:: none

    numba.typeinfer.TypingError: Failed at nopython (nopython frontend)
    Undeclared getitem(float64, int64)

.. code-block:: python

    a = 10.0
    a[0] = 2.0

* Tried to do `var[i]` where var is a float64, not an array of float64.
* Often happens due to confusion with array dimensions/scalars


Interpreting lowering errors
----------------------------

* Sometimes Numba produces weird errors if things slip through front-end checking
* This one is because broadcasting is not supported:

.. code-block:: none

    numba.lowering.LoweringError: Failed at nopython (nopython mode backend)
    Internal error:
    ValueError: '$0.22' is not a valid parameter name
    File "blackscholes.py", line 34

Try commenting out code until the error goes away to figure out the source.

Broadcasting/slicing error
--------------------------

Possibly due to an operation on two different sliced/broadcasted arrays:

.. code-block:: none

    raise LoweringError(msg, inst.loc)
    numba.lowering.LoweringError: Failed at nopython (nopython mode backend)
    Internal error:
    NotImplementedError: Don't know how to allocate array with layout 'A'.
    File "is_distance_solution.py", line 34


.. code-block:: none

    numba.typeinfer.TypingError: Failed at nopython (nopython frontend)
    Internal error at <numba.typeinfer.CallConstrain object at 0x7f1b3d9762e8>:
    Don't know how to create implicit output array with 'A' layout.
    File "pairwise.py", line 22


Treating array like a scalar
----------------------------

Another one, this time trying to check truth of an array:

.. code-block:: none

    Internal error:
    NotImplementedError: ('is_true', <llvmlite.ir.instructions.LoadInstr object at 0x7f2c311ff860>, array(bool, 1d, C))
    File "blackscholes_tutorial.py", line 26
    File "blackscholes_tutorial.py", line 45


Modes of compilation
--------------------

* *Nopython mode*: fastest mode, which all the restrictions apply to
* *Object mode*: supports all functions and types, but not much speedup
* For nopython mode:
  - Must be able to determine all types
  - All types and functions used must be supported
* Force nopython mode with `@jit(nopython=True)`

Tutorial exercise 2.2
=====================

`Compilation modes`


Loop lifting
------------

* In object mode, Numba attempts to extract loops and compile them in nopython mode.
* Good for functions bookended by nopython-unsupported code.

.. code-block:: python

    @jit
    def sum_strings(arr):
        intarr = np.empty(len(arr), dtype=np.int32)
        for i in range(len(arr)):
            intarr[i] = int(arr[i])
        sum = 0

        # Lifted loop
        for i in range(len(intarr)):
            sum += intarr[i]

         return sum


Tutorial Exercise 2.3
=====================

`Loop Lifting`


Using Numba "At Large"
======================


Tips 0 - Profiling
------------------

* Profiling is important
* You should only modify functions that take a significant amount of CPU time
* use cProfile then line_profiler
* gprof2dot handy for getting an overview

.. image:: /_static/gprof2dot.png


Tips 1 - General Approach
-------------------------

* Start off with just jitting it and see if it runs
* Use `numba --annotate-html` to see what Numba sees
* Start adding `nopython=True` to your innermost functions
* Try to fix each function and then move on

    - Need to make sure all inputs, outputs, are Numba-compatible types
    - No lists, dicts, etc

* Don't forget to assess performance at each state


Tips 2 - Don't Specify Types
----------------------------

* In the past Numba required you to specify types explicitly.
* Don't specify types unless absolutely necessary.
* Lots of examples on the web like this:

.. code-block:: python

    @jit(float64(float64, float64))
    def add(a, b):
        return a + b

* :code:`float64(float64, float64)` *probably unnecessary*!


Tips 3 - Optimisations
----------------------

.. code-block:: python

    for i in range(len(X)):
        Y[i] = sin(X[i])
    for i in range(len(Y)):
        Z[i] = Y[i] * Y[i]

1. Loop fusion:

.. code-block:: python

    for i in range(len(X)):
        Y[i] = sin(X[i])
        Z[i] = Y[i] * Y[i]

2. Array contraction:

.. code-block:: python

    for i in range(len(X)):
        Y = sin(X[i])
        Z[i] = Y * Y


Tips 4 - Debugging
------------------

* Numba is a bit like C - no bounds checking.
* Out of bounds writes can cause very odd behaviour!
* Set the env var ``NUMBA_DISABLE_JIT=1`` to disable compilation
* Then, Python checks may highlight problems


Tips 5 - Releasing the GIL
--------------------------

* N-core scalability by releasing the Global Interpreter Lock:

.. code-block:: python

    @numba.jit(nogil=True)
    def my_function(x, y, z):
        ...

* No protection from race conditions!
* Tip: use concurrent.futures.ThreadPoolExecutor on Python 3
* See ``examples/nogil.py`` in the Numba distribution


Example codes
-------------

* They all have timing and testing.
* Set up so you can modify one of its implementations to try and use Numba and go fast
* Some taken from examples, some found on the internet

    - see references in source

* Example solutions in the same folder


Example Optimisation Time
=========================

* Pick an example or some of your own code
* Use Numba to go as fast as possible


Future of Numba
---------------

Short- to medium-term roadmap:

- Python 3.5 support (already in Numba channel)
- Numpy 1.10 support (matmul ``@`` operator)
- ARMv7 support (Raspberry Pi 2)
- Parallel ufunc compilation (multicore CPUs and GPUs)
  ``@vectorize(target='cuda')``
- Jitting classes - struct-like objects with methods attached
- On-disk caching (minimise startup time)
- (Further in the future) distribution of compiled code (end user need not
  install Numba)


Blog posts
----------

* Numba and Cython - how to choose:
  http://stephanhoyer.com/2015/04/09/numba-vs-cython-how-to-choose/
* Ising model example:
  http://matthewrocklin.com/blog/work/2015/02/28/Ising/
* Playing with Numba and Finite Differences:
  http://nbviewer.ipython.org/gist/ketch/ae87a94f4ef0793d5d52


More info / contributing
------------------------

Repos, documentation, mailing list:

* https://github.com/numba/numba
* https://github.com/ContinuumIO/numbapro-examples
* http://numba.pydata.org/doc.html
* Numba-users mailing list

Commercial support: sales@continuum.io

* Consulting, enhancements, support for new architectures

* I will be around for the weekend + Monday sprints - come and talk!
